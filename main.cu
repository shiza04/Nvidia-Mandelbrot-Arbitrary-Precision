// Mandelbrot explorer – true 4096-bit fixed-point arithmetic on GPU
//
// Number format: Q4.4092  two's-complement, little-endian 64-bit words.
//   word[0]  = bits  0 .. 63   (least significant)
//   word[63] = bits 4032..4095  (most significant)
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
#define WORDS  64   // 64 × 64-bit words = 4096-bit number

struct Fixed4096 {
    unsigned long long data[WORDS];
};

// ═══════════════════════════════════════════════════════════════════════════
//  HOST helpers
// ═══════════════════════════════════════════════════════════════════════════

// Convert double to Q4.4092
static Fixed4096 doubleToFixed(double v) {
    Fixed4096 r;
    memset(r.data, 0, sizeof(r.data));

    bool negative = (v < 0.0);
    double av = negative ? -v : v;

    unsigned long long intPart = (unsigned long long)av;
    double frac = av - (double)intPart;

    // Shift integer part to the top 4 bits of the most significant word
    unsigned long long word63 = intPart << 60;

    // Extract the top 60 bits of the fraction into word63
    for (int b = 59; b >= 0; b--) {
        frac *= 2.0;
        if (frac >= 1.0) { word63 |= (1ULL << b); frac -= 1.0; }
    }
    r.data[WORDS - 1] = word63;

    // Extract remaining fractional bits
    for (int w = WORDS - 2; w >= 0; w--) {
        unsigned long long word = 0;
        for (int b = 63; b >= 0; b--) {
            frac *= 2.0;
            if (frac >= 1.0) { word |= (1ULL << b); frac -= 1.0; }
        }
        r.data[w] = word;
    }

    // Two's-complement negate if value was negative
    if (negative) {
        unsigned long long carry = 1;
        for (int i = 0; i < WORDS; i++) {
            unsigned long long s = ~r.data[i] + carry;
            carry = (s < (~r.data[i])) ? 1ULL : 0ULL;
            r.data[i] = s;
        }
    }
    return r;
}

// ═══════════════════════════════════════════════════════════════════════════
//  DEVICE arithmetic  (Q4.4092 two's-complement)
// ═══════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ Fixed4096 fp_add(const Fixed4096& a, const Fixed4096& b) {
    Fixed4096 res;
    unsigned long long carry = 0;
    for (int i = 0; i < WORDS; i++) {
        unsigned long long s  = a.data[i] + b.data[i];
        unsigned long long c1 = (s < a.data[i]) ? 1ULL : 0ULL;
        s += carry;
        unsigned long long c2 = (s < carry) ? 1ULL : 0ULL;
        res.data[i] = s;
        carry = c1 | c2;
    }
    return res;
}

__device__ __forceinline__ Fixed4096 fp_sub(const Fixed4096& a, const Fixed4096& b) {
    Fixed4096 res;
    unsigned long long borrow = 0;
    for (int i = 0; i < WORDS; i++) {
        unsigned long long ai = a.data[i];
        unsigned long long bi = b.data[i];
        unsigned long long d  = ai - bi - borrow;
        borrow = (ai < bi || (borrow && ai == bi)) ? 1ULL : 0ULL;
        res.data[i] = d;
    }
    return res;
}

__device__ __forceinline__ Fixed4096 fp_dbl(const Fixed4096& a) {
    Fixed4096 res;
    unsigned long long carry = 0;
    for (int i = 0; i < WORDS; i++) {
        res.data[i] = (a.data[i] << 1) | carry;
        carry = a.data[i] >> 63;
    }
    return res;
}

// Signed multiply: Q4.4092 × Q4.4092 → Q4.4092
__device__ Fixed4096 fp_mul(const Fixed4096& a, const Fixed4096& b) {
    bool negA = ((long long)a.data[WORDS - 1] < 0LL);
    bool negB = ((long long)b.data[WORDS - 1] < 0LL);
    bool negR = negA ^ negB;

    auto negate = [](const unsigned long long* src, unsigned long long* dst) {
        unsigned long long carry = 1;
        for (int i = 0; i < WORDS; i++) {
            unsigned long long s = ~src[i] + carry;
            carry = (s < (~src[i])) ? 1ULL : 0ULL;
            dst[i] = s;
        }
    };

    unsigned long long ma[WORDS], mb[WORDS];
    if (negA) negate(a.data, ma); else for (int i=0;i<WORDS;i++) ma[i]=a.data[i];
    if (negB) negate(b.data, mb); else for (int i=0;i<WORDS;i++) mb[i]=b.data[i];

    unsigned long long prod[2 * WORDS];
    for (int i = 0; i < 2 * WORDS; i++) prod[i] = 0ULL;

    for (int i = 0; i < WORDS; i++) {
        if (ma[i] == 0) continue;
        unsigned long long carry = 0;
        for (int j = 0; j < WORDS; j++) {
            unsigned long long lo  = ma[i] * mb[j];
            unsigned long long hi  = __umul64hi(ma[i], mb[j]);
            
            // Correct 3-operand addition with precise carry handling
            unsigned long long s1 = prod[i + j] + lo;
            unsigned long long c1 = (s1 < lo) ? 1ULL : 0ULL;
            unsigned long long sum = s1 + carry;
            unsigned long long c2 = (sum < carry) ? 1ULL : 0ULL;
            
            prod[i + j] = sum;
            carry = hi + c1 + c2;
        }
        prod[i + WORDS] += carry;
    }

    // Right-shift 4092 bits for Q4.4092 alignment
    Fixed4096 res;
    for (int k = 0; k < WORDS; k++) {
        res.data[k] = (prod[k + (WORDS-1)] >> 60) | (prod[k + WORDS] << 4);
    }

    if (negR) {
        unsigned long long carry = 1;
        for (int i = 0; i < WORDS; i++) {
            unsigned long long s = ~res.data[i] + carry;
            carry = (s < (~res.data[i])) ? 1ULL : 0ULL;
            res.data[i] = s;
        }
    }
    return res;
}

// Escape test: zx² + zy² >= 4.0
// In Q4.4092, 4.0 sets bit 62 of the highest word.
__device__ __forceinline__ bool fp_escaped(const Fixed4096& zx2, const Fixed4096& zy2) {
    unsigned long long carry = 0;
    unsigned long long top   = 0;
    for (int i = 0; i < WORDS; i++) {
        unsigned long long s  = zx2.data[i] + zy2.data[i];
        unsigned long long c1 = (s < zx2.data[i]) ? 1ULL : 0ULL;
        s += carry;
        unsigned long long c2 = (s < carry) ? 1ULL : 0ULL;
        carry = c1 | c2;
        if (i == WORDS - 1) top = s;
    }
    return carry || ((top >> 62) != 0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Kernel
// ═══════════════════════════════════════════════════════════════════════════
__global__ void mandelKernel(
    uchar4* out,
    const Fixed4096* __restrict__ d_cx,
    const Fixed4096* __restrict__ d_cy,
    int w, int h, int maxIter)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= w || py >= h) return;

    int idx = py * w + px;
    Fixed4096 cx = d_cx[idx];
    Fixed4096 cy = d_cy[idx];

    Fixed4096 zx, zy;
    for (int k = 0; k < WORDS; k++) { zx.data[k] = 0ULL; zy.data[k] = 0ULL; }

    int i = 0;
    for (; i < maxIter; i++) {
        Fixed4096 zx2 = fp_mul(zx, zx);
        Fixed4096 zy2 = fp_mul(zy, zy);

        if (fp_escaped(zx2, zy2)) break;

        Fixed4096 new_zy = fp_add(fp_dbl(fp_mul(zx, zy)), cy);
        Fixed4096 new_zx = fp_add(fp_sub(zx2, zy2), cx);

        zx = new_zx;
        zy = new_zy;
    }

    float t = (float)i / maxIter;
    out[idx] = make_uchar4(
        (unsigned char)(8.5f  * (1-t)*t*t*t             * 255),
        (unsigned char)(15.0f * (1-t)*(1-t)*t*t         * 255),
        (unsigned char)(9.0f  * (1-t)*(1-t)*(1-t)*t     * 255),
        255
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  Host: build per-pixel cx / cy coordinate arrays
// ═══════════════════════════════════════════════════════════════════════════
static void buildCoords(
    std::vector<Fixed4096>& h_cx,
    std::vector<Fixed4096>& h_cy,
    int w, int h, double zoom, double offX, double offY)
{
    h_cx.resize((size_t)w * h);
    h_cy.resize((size_t)w * h);

    double scale = 4.0 / (w * zoom);

    // Determine number of hardware threads available
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Fallback

    std::vector<std::thread> workers;
    
    // Lambda to process a range of rows
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

    // Divide the work (rows) among threads
    int rowsPerThread = h / numThreads;
    for (int t = 0; t < numThreads; t++) {
        int startY = t * rowsPerThread;
        int endY = (t == numThreads - 1) ? h : (t + 1) * rowsPerThread;
        workers.emplace_back(workerTask, startY, endY);
    }

    // Wait for all threads to finish
    for (auto& thread : workers) {
        thread.join();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    if (!glfwInit()) return -1;

    const int W = 100, H = 75;
    GLFWwindow* window = glfwCreateWindow(W, H, "Mandelbrot 4096-bit CUDA", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; return -1; }

    uchar4* d_out; cudaMalloc(&d_out, W * H * sizeof(uchar4));
    Fixed4096* d_cx;  cudaMalloc(&d_cx,  (size_t)W * H * sizeof(Fixed4096));
    Fixed4096* d_cy;  cudaMalloc(&d_cy,  (size_t)W * H * sizeof(Fixed4096));

    std::vector<uchar4>    h_out(W * H);
    std::vector<Fixed4096> h_cx, h_cy;

    double zoom  =  1.0;
    double offX  = -0.5;
    double offY  =  0.0;
    int    maxIter = 256;

    double prev_zoom = -1, prev_offX = 1e300, prev_offY = 1e300;
    int    prev_iter = -1;

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
            cudaMemcpy(d_cx, h_cx.data(), (size_t)W*H*sizeof(Fixed4096), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cy, h_cy.data(), (size_t)W*H*sizeof(Fixed4096), cudaMemcpyHostToDevice);
            prev_zoom = zoom; prev_offX = offX; prev_offY = offY; prev_iter = maxIter;
        }

        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);
        mandelKernel<<<grid, block>>>(d_out, d_cx, d_cy, W, H, maxIter);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out.data(), d_out, W * H * sizeof(uchar4), cudaMemcpyDeviceToHost);
        glDrawPixels(W, H, GL_RGBA, GL_UNSIGNED_BYTE, h_out.data());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(d_out);
    cudaFree(d_cx);
    cudaFree(d_cy);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}