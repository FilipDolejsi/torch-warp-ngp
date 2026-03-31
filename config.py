import math

class Config:
    # Data Paths
    DATA_ROOT = "../instant-ngp/data/lego"
    
    # Hash Encoding (Paper Section 3)
    L = 16              # Number of levels
    F = 2               # Feature dimension per level
    
    # Paper Default (Table 1 & Fig 5): T=2^19 is the recommended trade-off
    T = 2**19           # Hash table size (524288)
    
    N_MIN = 16          # Coarsest resolution
    N_MAX = 2048        # Finest resolution (Sec 5.4: "set to 2048 ... for NeRF")
    
    # Architecture
    HIDDEN_DIM_DENSITY = 64
    HIDDEN_DIM_COLOR = 64
    
    # Hashing Primes (Eq. 4)
    PRIME_1 = 1
    PRIME_2 = 2654435761
    PRIME_3 = 805459861

    # Training (Paper Section 4 & 5.4)
    BATCH_SIZE = 4096  
    LR = 1e-2
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.99
    ADAM_EPS = 1e-15    
    
    ITERATIONS = 100000 
    VAL_INTERVAL = 2000 
    
    # Rendering & Transparency Handling
    N_SAMPLES = 512      
    
    RANDOM_BG_TRAIN = True 
    
    # Scene Bounds
    AABB_MIN = [-1.5, -1.5, -1.5]
    AABB_MAX = [ 1.5,  1.5,  1.5]
    
    # Toggle Backend
    USE_WARP = False    

    DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
