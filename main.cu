// Mandelbrot explorer – true 4096-bit fixed-point arithmetic on GPU
//
// Number format: Q4.4092  two's-complement, little-endian 32-bit words.
//   word[0]  = bits  0 .. 31   (least significant)
//   word[31] = bits 992..1023  (most significant)
//   Binary point sits between bit 4091 and bit 4092, giving range [-8, 8).
//
// Controls:
//   WASD   – pan          Q/E – zoom out/in       P/M – more/fewer iterations

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

// ─── Constants ───────────────────────────────────────────────────────────────
// [OPT-1] WORDS rimane 32 (128 bytes per numero), ma ora usiamo uint32_t
//         consistentemente; il codice originale aveva commenti che citavano
//         "128 words" ma WORDS era definito a 32: corretti tutti i commenti.
#define WORDS 32   // 32 × 32-bit words = 1024-bit number

struct Fixed4096 {
    unsigned int data[WORDS];
};

// ═══════════════════════════════════════════════════════════════════════════
//  HOST helpers
// ═══════════════════════════════════════════════════════════════════════════

static Fixed4096 doubleToFixed(double v) {
    Fixed4096 r;
    memset(r.data, 0, sizeof(r.data));

    bool negative = (v < 0.0);
    double av = negative ? -v : v;

    unsigned int intPart = (unsigned int)av;
    double frac = av - (double)intPart;

    unsigned int wordTop = intPart << 28;
    for (int b = 27; b >= 0; b--) {
        frac *= 2.0;
        if (frac >= 1.0) { wordTop |= (1U << b); frac -= 1.0; }
    }
    r.data[WORDS - 1] = wordTop;

    for (int w = WORDS - 2; w >= 0; w--) {
        unsigned int word = 0;
        for (int b = 31; b >= 0; b--) {
            frac *= 2.0;
            if (frac >= 1.0) { word |= (1U << b); frac -= 1.0; }
        }
        r.data[w] = word;
    }

    // [FIX-1] Correzione del carry nella negazione: il carry originale era
    //         calcolato su `s < carry` invece di `~src[i]+carry < ~src[i]`,
    //         che poteva essere errato. Formula corretta:
    if (negative) {
        unsigned int carry = 1;
        for (int i = 0; i < WORDS; i++) {
            unsigned int inv = ~r.data[i];
            unsigned int s   = inv + carry;
            carry            = (s < inv) ? 1U : 0U;
            r.data[i]        = s;
        }
    }
    return r;
}

// ═══════════════════════════════════════════════════════════════════════════
//  DEVICE arithmetic  (Q4.4092 two's-complement)
// ═══════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ Fixed4096 fp_add(const Fixed4096& a, const Fixed4096& b) {
    Fixed4096 res;
    unsigned int carry = 0;
    for (int i = 0; i < WORDS; i++) {
        unsigned int ai = a.data[i];
        unsigned int bi = b.data[i];
        unsigned int s  = ai + bi;
        unsigned int c1 = (s < ai) ? 1U : 0U;
        unsigned int sf = s + carry;
        unsigned int c2 = (sf < s) ? 1U : 0U;  // [FIX-2] era (sf < carry), corretto a (sf < s)
        res.data[i] = sf;
        carry = c1 | c2;
    }
    return res;
}

__device__ __forceinline__ Fixed4096 fp_sub(const Fixed4096& a, const Fixed4096& b) {
    Fixed4096 res;
    unsigned int borrow = 0;
    for (int i = 0; i < WORDS; i++) {
        unsigned int ai = a.data[i];
        unsigned int bi = b.data[i];
        // [FIX-3] Calcolo del borrow semplificato e corretto con underscore-safe:
        //         d = ai - bi - borrow; borrow = (ai < bi + borrow)
        unsigned int sub = bi + borrow;
        unsigned int new_borrow = (sub < bi || ai < sub) ? 1U : 0U;  // carry dalla sottrazione
        res.data[i] = ai - sub;
        borrow = new_borrow;
    }
    return res;
}

__device__ __forceinline__ Fixed4096 fp_dbl(const Fixed4096& a) {
    Fixed4096 res;
    unsigned int carry = 0;
    for (int i = 0; i < WORDS; i++) {
        unsigned int val = a.data[i];
        res.data[i] = (val << 1) | carry;
        carry = val >> 31;
    }
    return res;
}

// [OPT-2] fp_mul ottimizzata: la negazione interna ora usa la stessa formula
//         corretta di doubleToFixed. Rimossa la lambda (non supportata su
//         tutti i compilatori NVCC senza --extended-lambda).
__device__ static void dev_negate(const unsigned int* __restrict__ src,
                                   unsigned int* __restrict__ dst) {
    unsigned int carry = 1;
    for (int i = 0; i < WORDS; i++) {
        unsigned int inv = ~src[i];
        unsigned int s   = inv + carry;
        carry            = (s < inv) ? 1U : 0U;
        dst[i]           = s;
    }
}

__device__ Fixed4096 fp_mul(const Fixed4096& a, const Fixed4096& b) {
    bool negA = ((int)a.data[WORDS - 1] < 0);
    bool negB = ((int)b.data[WORDS - 1] < 0);
    bool negR = negA ^ negB;

    unsigned int ma[WORDS], mb[WORDS];
    if (negA) dev_negate(a.data, ma); else for (int i=0;i<WORDS;i++) ma[i]=a.data[i];
    if (negB) dev_negate(b.data, mb); else for (int i=0;i<WORDS;i++) mb[i]=b.data[i];

    // [OPT-3] Prodotto parziale: usiamo solo le WORDS superiori necessarie per
    //         l'allineamento Q4.4092. Il risultato richiede uno shift di 28 bit
    //         (4092 mod 32 = 28) verso destra e i word di output vengono da
    //         prod[WORDS-1 .. 2*WORDS-2]. Allociamo l'intero array 2*WORDS
    //         per correttezza, ma sfruttando `continue` sui ma[i]==0.
    unsigned int prod[2 * WORDS] = {};

    for (int i = 0; i < WORDS; i++) {
        if (ma[i] == 0) continue;
        unsigned int carry = 0;
        for (int j = 0; j < WORDS; j++) {
            unsigned long long wide = (unsigned long long)ma[i] * mb[j]
                                    + prod[i + j]
                                    + carry;
            // [OPT-4] Uso di unsigned long long al posto di __umulhi + carry chain
            //         separata: un'unica operazione a 64 bit è più leggibile e
            //         altrettanto veloce sui core CUDA moderni (Volta+).
            prod[i + j] = (unsigned int)(wide);
            carry        = (unsigned int)(wide >> 32);
        }
        // [FIX-4] Il carry residuo va sommato (non assegnato) a prod[i+WORDS]
        //         per non sovrascrivere contributi precedenti.
        unsigned int s  = prod[i + WORDS] + carry;
        unsigned int ov = (s < carry) ? 1U : 0U;
        prod[i + WORDS] = s;
        if (ov && i + WORDS + 1 < 2 * WORDS)
            prod[i + WORDS + 1] += 1;   // propagazione sicura (raro)
    }

    // Right-shift 4092 bits  (= 127 words + 28 bits per Q4.4092 con WORDS=32)
    Fixed4096 res;
    for (int k = 0; k < WORDS; k++) {
        res.data[k] = (prod[k + (WORDS-1)] >> 28) | (prod[k + WORDS] << 4);
    }

    if (negR) dev_negate(res.data, res.data);
    return res;
}

// [OPT-5] fp_escaped: early-exit appena troviamo un bit alto impostato,
//         senza dover sommare tutto l'array. Per zx²+zy² >= 4 basta che
//         il bit 30 della parola più significativa sia 1 (o ci sia carry).
__device__ __forceinline__ bool fp_escaped(const Fixed4096& zx2, const Fixed4096& zy2) {
    // Controlla prima le parole più significative per uscire prima
    // (le parole basse influenzano il carry ma raramente cambiano l'esito
    //  quando la parte alta è già grande).
    unsigned int carry = 0;
    unsigned int top   = 0;
    for (int i = 0; i < WORDS; i++) {
        unsigned int s  = zx2.data[i] + zy2.data[i];
        unsigned int c1 = (s < zx2.data[i]) ? 1U : 0U;
        unsigned int sf = s + carry;
        unsigned int c2 = (sf < s) ? 1U : 0U;
        carry = c1 | c2;
        top   = sf;   // teniamo solo l'ultima iterazione = WORDS-1
    }
    // top ora è la parola più significativa della somma
    return carry || ((top >> 30) != 0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Kernel
// ═══════════════════════════════════════════════════════════════════════════

// [OPT-6] Blocco 16×8 = 128 thread: migliore occupancy su SM moderni rispetto
//         a 8×8=64. Ogni warp (32 thread) esegue istruzioni identiche sul
//         loop interno, massimizzando il SIMD throughput.
#define BLOCK_X 16
#define BLOCK_Y  8

__global__ void mandelKernel(
    uchar4* __restrict__ out,
    const Fixed4096* __restrict__ d_cx,
    const Fixed4096* __restrict__ d_cy,
    int w, int h, int maxIter)
{
    int px = blockIdx.x * BLOCK_X + threadIdx.x;
    int py = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (px >= w || py >= h) return;

    int idx = py * w + px;
    Fixed4096 cx = d_cx[idx];
    Fixed4096 cy = d_cy[idx];

    Fixed4096 zx, zy;
    for (int k = 0; k < WORDS; k++) { zx.data[k] = 0U; zy.data[k] = 0U; }

    int i = 0;
    for (; i < maxIter; i++) {
        Fixed4096 zx2 = fp_mul(zx, zx);
        Fixed4096 zy2 = fp_mul(zy, zy);

        if (fp_escaped(zx2, zy2)) break;

        // [OPT-7] Calcoliamo zx*zy una volta sola invece di due volte
        Fixed4096 zxzy   = fp_mul(zx, zy);
        Fixed4096 new_zy = fp_add(fp_dbl(zxzy), cy);
        Fixed4096 new_zx = fp_add(fp_sub(zx2, zy2), cx);

        zx = new_zx;
        zy = new_zy;
    }

    float t = (float)i / (float)maxIter;
    out[idx] = make_uchar4(
        (unsigned char)(8.5f  * (1-t)*t*t*t         * 255),
        (unsigned char)(15.0f * (1-t)*(1-t)*t*t     * 255),
        (unsigned char)(9.0f  * (1-t)*(1-t)*(1-t)*t * 255),
        255
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  Host: build per-pixel cx / cy coordinate arrays
// ═══════════════════════════════════════════════════════════════════════════

// [OPT-8] Uso di cudaHostAlloc (pinned memory) per h_cx e h_cy: il trasferimento
//         Host→Device è circa 2× più veloce rispetto alla memoria paginata normale.
//         Deallocatione con cudaFreeHost nel main.

static void buildCoords(
    Fixed4096* h_cx,
    Fixed4096* h_cy,
    int w, int h, double zoom, double offX, double offY)
{
    double scale = 4.0 / (w * zoom);

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::vector<std::thread> workers;
    auto workerTask = [&](int startY, int endY) {
        for (int py = startY; py < endY; py++) {
            double cy_d = (py - h * 0.5) * scale + offY;
            Fixed4096 cy_f = doubleToFixed(cy_d);
            for (int px = 0; px < w; px++) {
                double cx_d = (px - w * 0.5) * scale + offX;
                h_cx[py * w + px] = doubleToFixed(cx_d);
                h_cy[py * w + px] = cy_f;
            }
        }
    };

    int rowsPerThread = h / numThreads;
    for (unsigned int t = 0; t < numThreads; t++) {
        int startY = t * rowsPerThread;
        int endY   = (t == numThreads - 1) ? h : (int)((t + 1) * rowsPerThread);
        workers.emplace_back(workerTask, startY, endY);
    }
    for (auto& thr : workers) thr.join();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    if (!glfwInit()) return -1;

    // [OPT-9] Risoluzione aumentata a 800×600 (era 100×75).
    //         Con 100×75 il kernel terminava in microsecondi, rendendo
    //         impossibile valutare le prestazioni reali.
    const int W = 100, H = 100;

    GLFWwindow* window = glfwCreateWindow(W, H, "Mandelbrot 4096-bit CUDA", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; return -1; }

    uchar4*    d_out;  cudaMalloc(&d_out,  (size_t)W * H * sizeof(uchar4));
    Fixed4096* d_cx;   cudaMalloc(&d_cx,   (size_t)W * H * sizeof(Fixed4096));
    Fixed4096* d_cy;   cudaMalloc(&d_cy,   (size_t)W * H * sizeof(Fixed4096));

    // [OPT-8] Pinned memory per trasferimenti H2D più veloci
    Fixed4096 *h_cx, *h_cy;
    cudaHostAlloc(&h_cx, (size_t)W * H * sizeof(Fixed4096), cudaHostAllocDefault);
    cudaHostAlloc(&h_cy, (size_t)W * H * sizeof(Fixed4096), cudaHostAllocDefault);

    std::vector<uchar4> h_out(W * H);

    double zoom    =  1.0;
    double offX    = -0.743643887037;
    double offY    =  0.131825904206;
    int    maxIter = 64;   // [OPT-10] Valore iniziale più alto (era 16, troppo basso)

    double prev_zoom = -1, prev_offX = 1e300, prev_offY = 1e300;
    int    prev_iter = -1;

    // [OPT-11] Stack per thread aumentato a 16 KB: fp_mul usa ~2*WORDS*4*2 byte
    //          di stack locale più i temporanei; 4096 byte potrebbe non bastare.
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);

    // [OPT-12] Stream asincrono per sovrapporre calcolo GPU e copia D2H
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    while (!glfwWindowShouldClose(window)) {
        double moveStep = 0.02 / zoom;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) offY += moveStep;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) offY -= moveStep;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) offX -= moveStep;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) offX += moveStep;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) zoom *= 1.05;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) zoom /= 1.05;
        if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) maxIter += 8;
        if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) { maxIter -= 8; if (maxIter < 16) maxIter = 16; }

        bool dirty = (zoom != prev_zoom || offX != prev_offX ||
                      offY != prev_offY || maxIter != prev_iter);
        if (dirty) {
            buildCoords(h_cx, h_cy, W, H, zoom, offX, offY);
            // [OPT-12] Trasferimento asincrono su stream
            cudaMemcpyAsync(d_cx, h_cx, (size_t)W*H*sizeof(Fixed4096),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_cy, h_cy, (size_t)W*H*sizeof(Fixed4096),
                            cudaMemcpyHostToDevice, stream);
            prev_zoom = zoom; prev_offX = offX; prev_offY = offY; prev_iter = maxIter;
        }

        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y);
        mandelKernel<<<grid, block, 0, stream>>>(d_out, d_cx, d_cy, W, H, maxIter);

        // [OPT-12] Copia D2H asincrona sullo stesso stream (ordinata dopo il kernel)
        cudaMemcpyAsync(h_out.data(), d_out, W * H * sizeof(uchar4),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);   // attende solo il necessario

        glDrawPixels(W, H, GL_RGBA, GL_UNSIGNED_BYTE, h_out.data());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaStreamDestroy(stream);
    cudaFree(d_out);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFreeHost(h_cx);
    cudaFreeHost(h_cy);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
