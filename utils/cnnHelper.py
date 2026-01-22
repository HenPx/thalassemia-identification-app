import numpy as np
# -----------------------
# im2col / col2im (Conv 2D)
# -----------------------
def get_out_dims(H, W, kH, kW, stride, pad):
    # Menghitung ukuran output conv (height & width)
    H_out = (H + 2*pad - kH)//stride + 1
    W_out = (W + 2*pad - kW)//stride + 1
    return H_out, W_out

def im2col_indices(x, kH, kW, stride=1, pad=0):
    # x: (N,H,W,C)
    # Mengubah patch-patch (kH×kW×C) dari image menjadi kolom
    # agar operasi convolution bisa dijadikan matrix multiplication (lebih cepat)
    N, H, W, C = x.shape
    H_out, W_out = get_out_dims(H, W, kH, kW, stride, pad)
    x_padded = np.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='constant')
    # Indeks kernel
    i0 = np.repeat(np.arange(kH), kW)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(kW), kH*C)
    j1 = stride * np.tile(np.arange(W_out), H_out)
    i = i0.reshape(-1,1) + i1.reshape(1,-1)
    j = j0.reshape(-1,1) + j1.reshape(1,-1)
    c = np.repeat(np.arange(C), kH*kW).reshape(-1,1)
    cols = x_padded[:, i, j, c]  # (N, kH*kW*C, H_out*W_out)
    cols = cols.transpose(1,2,0).reshape(kH*kW*C, -1)  # (K, N*H_out*W_out)
    return cols

def col2im_indices(cols, x_shape, kH, kW, stride=1, pad=0):
    # Kebalikan dari im2col: mengubah hasil gradient bentuk kolom
    # kembali menjadi bentuk gambar (dx)
    N, H, W, C = x_shape
    H_out, W_out = get_out_dims(H, W, kH, kW, stride, pad)
    x_padded = np.zeros((N, H + 2*pad, W + 2*pad, C), dtype=cols.dtype)
    i0 = np.repeat(np.arange(kH), kW)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(kW), kH*C)
    j1 = stride * np.tile(np.arange(W_out), H_out)
    i = i0.reshape(-1,1) + i1.reshape(1,-1)
    j = j0.reshape(-1,1) + j1.reshape(1,-1)
    c = np.repeat(np.arange(C), kH*kW).reshape(-1,1)

    cols_reshaped = cols.reshape(kH*kW*C, H_out*W_out, N).transpose(2,0,1)
    np.add.at(x_padded, (slice(None), i, j, c), cols_reshaped)
    if pad == 0:
        return x_padded
    return x_padded[:, pad:-pad, pad:-pad, :]

# -----------------------
# Layer: Conv2D (im2col)
# -----------------------

class Conv2D:
    # Tambahkan parameter 'weights' dan 'trainable'
    def __init__(self, in_ch, out_ch, kH=3, kW=3, stride=1, pad=1, weights=None, trainable=True):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kH = kH
        self.kW = kW
        self.stride = stride
        self.pad = pad
        self.trainable = trainable # Flag untuk freeze layer

        if weights is not None:
            # Jika bobot Gabor diberikan, pakai itu
            self.W = weights
            # Pastikan shape sesuai: (out_ch, in_ch, kH, kW)
        else:
            # He init standar
            fan_in = in_ch * kH * kW
            self.W = np.random.randn(out_ch, in_ch, kH, kW).astype(np.float32) * np.sqrt(2.0 / fan_in)

        self.b = np.zeros((out_ch,), dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache = None

    def forward(self, x):
        self.x_shape = x.shape
        N, H, W, C = x.shape
        cols = im2col_indices(x, self.kH, self.kW, self.stride, self.pad)
        W_col = self.W.reshape(self.out_ch, -1)
        out = (W_col @ cols + self.b[:, None]).astype(np.float32)
        H_out, W_out = get_out_dims(H, W, self.kH, self.kW, self.stride, self.pad)
        out = out.reshape(self.out_ch, H_out, W_out, N).transpose(3,1,2,0)
        self.cache = (cols, W_col)
        return out

    def backward(self, dout):
        cols, W_col = self.cache
        N = dout.shape[0]
        dout_reshaped = dout.transpose(3,1,2,0).reshape(self.out_ch, -1)
        self.db = np.sum(dout_reshaped, axis=1).astype(np.float32)
        self.dW = (dout_reshaped @ cols.T).reshape(self.W.shape).astype(np.float32)
        dcols = (W_col.T @ dout_reshaped).astype(np.float32)
        dx = col2im_indices(dcols, self.x_shape, self.kH, self.kW, self.stride, self.pad).astype(np.float32)
        return dx

    def step(self, lr):
        # --- PERUBAHAN PENTING DISINI ---
        if not self.trainable:
            return # Skip update jika layer dibekukan (Gabor)

        self.W -= lr * self.dW
        self.b -= lr * self.db
        
        
# -----------------------
# Layer: ReLU
# -----------------------
class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask
    def backward(self, dout):
        return dout * self.mask
    def step(self, lr):
        pass
    
    
# -----------------------
# Layer: MaxPool2x2
# -----------------------
class MaxPool2x2:
    def __init__(self):
        self.cache = None
    def forward(self, x):
        # x: (N,H,W,C), pool 2x2 stride 2
        N, H, W, C = x.shape
        H_out = H // 2
        W_out = W // 2
        x_resh = x.reshape(N, H_out, 2, W_out, 2, C)
        out = x_resh.max(axis=(2,4))
        # mask untuk backward
        self.cache = (x, x_resh, out)
        return out
    def backward(self, dout):
        x, x_resh, out = self.cache
        N, H, W, C = x.shape
        H_out = H // 2
        W_out = W // 2
        # broadcast mask
        out_expanded = out[:, :, None, :, None, :]
        mask = (x_resh == out_expanded).astype(np.float32)
        d_in = np.zeros_like(x_resh, dtype=np.float32)
        d_in += dout[:, :, None, :, None, :] * mask
        dx = d_in.reshape(N, H, W, C).astype(np.float32)
        return dx
    def step(self, lr):
        pass
    
    
# -----------------------
# Layer: Flatten
# -----------------------
class Flatten:
    def __init__(self):
        self.shape = None
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, dout):
        return dout.reshape(self.shape)
    def step(self, lr):
        pass


# -----------------------
# Layer: Dense (FC)
# -----------------------
class Dense:
    def __init__(self, in_dim, out_dim):
        # He-like for linear input
        self.W = (np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0/in_dim)).astype(np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dout):
        self.dW = (self.x.T @ dout).astype(np.float32)
        self.db = np.sum(dout, axis=0).astype(np.float32)
        dx = (dout @ self.W.T).astype(np.float32)
        return dx
    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


# -----------------------
# Softmax + Cross Entropy
# -----------------------
def softmax_logits(z):
    # z: (N,C)
    zmax = np.max(z, axis=1, keepdims=True)
    e = np.exp((z - zmax).astype(np.float32))
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy_loss(logits, y):
    # logits: (N,C), y: (N,)
    probs = softmax_logits(logits)
    N = logits.shape[0]
    logp = -np.log(probs[np.arange(N), y] + 1e-12)
    loss = np.mean(logp).astype(np.float32)
    return loss, probs

def softmax_ce_backward(probs, y):
    N = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(N), y] -= 1.0
    grad = grad / N
    return grad.astype(np.float32)



class ModerateCNN_GABOR_COMBINED:
    def __init__(self, num_classes):

        # Layer 0: Gabor (Fixed)
        # Input: 1 channel, Output: 4 channel
        self.layer_gabor = None
        self.relu_gabor = ReLU()

        # Sisa Layer (CNN Biasa yang bisa belajar)
        self.layers = [
            # --- BLOK 2 ---
            # PERHATIKAN: in_ch = 5
            # (4 dari Gabor + 1 dari Gambar Asli)
            # --- BLOCK 1 (Output Channel: 32) ---
            Conv2D(in_ch=5, out_ch=32, kH=3, kW=3, stride=1, pad=1),
            ReLU(),
            MaxPool2x2(),            # Output Size: 32x32x32

            # --- BLOCK 2 (Output Channel: 64) ---
            Conv2D(in_ch=32, out_ch=64, kH=3, kW=3, stride=1, pad=1),
            ReLU(),
            MaxPool2x2(),            # Output Size: 16x16x64

            # --- BLOCK 3 (Output Channel: 128) ---
            Conv2D(in_ch=64, out_ch=128, kH=3, kW=3, stride=1, pad=1),
            ReLU(),
            MaxPool2x2(),            # Output Size: 8x8x128

            # --- FLATTEN & DENSE ---
            Flatten(),

            # PERHITUNGAN BARU UNTUK 64x64:
            # Tinggi(8) * Lebar(8) * Channel(128) = 8192
            Dense(8*8*128, num_classes)
        ]
    
    def forward(self, x):
        # 1. Simpan gambar asli
        x_original = x

        # 2. Proses Gabor
        out_gabor = self.layer_gabor.forward(x)
        out_gabor = self.relu_gabor.forward(out_gabor)

        # 3. PENGGABUNGAN (CONCATENATION)
        # axis=3 adalah axis channel (N, H, W, C)
        # Gabung [Output Gabor (4 ch), Gambar Asli (1 ch)] -> Total 5 ch
        x_combined = np.concatenate((out_gabor, x_original), axis=3)

        # 4. Lanjutkan ke layer berikutnya dengan data gabungan
        out = x_combined
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, dout):
        # Alur backward juga harus manual sedikit di awal

        # 1. Backward untuk layer-layer biasa (Block 3 s/d Block 2)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        # Sekarang 'dout' bentuknya (N, H, W, 5) karena tadi inputnya 5 channel
        # Kita harus PECAH lagi gradient-nya:
        # - 4 channel pertama milik Gabor
        # - 1 channel terakhir milik input asli (skip connection)

        dout_gabor = dout[:, :, :, :4] # Ambil 4 channel awal
        dout_original_skip = dout[:, :, :, 4:] # Ambil 1 channel akhir

        # 2. Backward ke layer Gabor
        # (Sebenarnya tidak ngefek ke bobot karena trainable=False,
        # tapi perlu untuk mengalirkan gradient ke input pixel)
        dout_from_gabor_layer = self.relu_gabor.backward(dout_gabor)
        dout_from_gabor_layer = self.layer_gabor.backward(dout_from_gabor_layer)

        # 3. Total Gradient ke Input
        # Gradient datang dari dua jalur: jalur gabor + jalur langsung (original)
        dout_final = dout_from_gabor_layer + dout_original_skip

        return dout_final

    def step(self, lr):
        # Update layer gabor (akan di-skip di dalam karena trainable=False)
        self.layer_gabor.step(lr)

        # Update layer lainnya
        for layer in self.layers:
            if hasattr(layer, 'step'):
                layer.step(lr)