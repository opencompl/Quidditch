
.text
.globl streamerSetup3
.p2align 2
streamerSetup3:
    li t0, 49
    scfgwi t0, 66
    li t0, 8
    scfgwi t0, 194
    scfgwi zero, 34
    li t0, 0x1000e780
    scfgwi t0, 898
    ret

.text
.globl kernel161
.p2align 2
kernel161:
    mv t2, a0
    mv t1, a1
    mv t0, a2
    li t3, 160
    scfgwi t3, 64 # dm 0 dim 0 bound
    li t3, 8
    scfgwi t3, 192 # dm 0 dim 0 stride
    li t3, 4
    scfgwi t3, 32 # dm 0 repeat
    li t3, 4
    scfgwi t3, 65 # dm 1 dim 0 bound
    li t3, 160
    scfgwi t3, 97 # dm 1 dim 1 bound
    li t3, 1288
    scfgwi t3, 193 # dm 1 dim 0 stride
    li t3, -5144
    scfgwi t3, 225 # dm 1 dim 1 stride
    scfgwi zero, 33 # dm 1 repeat
    scfgwi t2, 768  # dm 0 dim 0 source
    scfgwi t1, 801  # dm 1 dim 1 source
    csrrsi zero, 1984, 1 # SSR enable
    mv t1, t0
    fld ft7, 0(t1) # load double from memref of shape (1, 5)
    fld ft6, 8(t0) # load double from memref of shape (1, 5)
    fld ft5, 16(t0) # load double from memref of shape (1, 5)
    fld ft4, 24(t0) # load double from memref of shape (1, 5)
    fld ft3, 32(t0) # load double from memref of shape (1, 5)
    li t1, 160
    frep.o t1, 5, 0, 0
    fmadd.d ft7, ft0, ft1, ft7
    fmadd.d ft6, ft0, ft1, ft6
    fmadd.d ft5, ft0, ft1, ft5
    fmadd.d ft4, ft0, ft1, ft4
    fmadd.d ft3, ft0, ft1, ft3
    mv t1, t0
    fsd ft7, 0(t1) # store double value to memref of shape (1, 5)
    fsd ft6, 8(t0) # store double value to memref of shape (1, 5)
    fsd ft5, 16(t0) # store double value to memref of shape (1, 5)
    fsd ft4, 24(t0) # store double value to memref of shape (1, 5)
    fsd ft3, 32(t0) # store double value to memref of shape (1, 5)
    csrrci zero, 1984, 1 # SSR disable
    ret


.text
.globl kernel100
.p2align 2
kernel100:
    mv t2, a0
    mv t1, a1
    mv t0, a2
    li t3, 99
    scfgwi t3, 64 # dm 0 dim 0 bound
    li t3, 8
    scfgwi t3, 192 # dm 0 dim 0 stride
    li t3, 4
    scfgwi t3, 32 # dm 0 repeat
    li t3, 4
    scfgwi t3, 65 # dm 1 dim 0 bound
    li t3, 99
    scfgwi t3, 97 # dm 1 dim 1 bound
    li t3, 800
    scfgwi t3, 193 # dm 1 dim 0 stride
    li t3, -3192
    scfgwi t3, 225 # dm 1 dim 1 stride
    scfgwi zero, 33 # dm 1 repeat
    scfgwi t2, 768  # dm 0 dim 0 source
    scfgwi t1, 801  # dm 1 dim 1 source
    csrr x0, 0x7C2
    csrrsi zero, 1984, 1 # SSR enable
    mv t1, t0
    fld ft7, 0(t1) # load double from memref of shape (1, 5)
    fld ft6, 8(t0) # load double from memref of shape (1, 5)
    fld ft5, 16(t0) # load double from memref of shape (1, 5)
    fld ft4, 24(t0) # load double from memref of shape (1, 5)
    fld ft3, 32(t0) # load double from memref of shape (1, 5)
    li t1, 99
    frep.o t1, 5, 0, 0
    fmadd.d ft7, ft0, ft1, ft7
    fmadd.d ft6, ft0, ft1, ft6
    fmadd.d ft5, ft0, ft1, ft5
    fmadd.d ft4, ft0, ft1, ft4
    fmadd.d ft3, ft0, ft1, ft3
    mv t1, t0
    fsd ft7, 0(t1) # store double value to memref of shape (1, 5)
    fsd ft6, 8(t0) # store double value to memref of shape (1, 5)
    fsd ft5, 16(t0) # store double value to memref of shape (1, 5)
    fsd ft4, 24(t0) # store double value to memref of shape (1, 5)
    fsd ft3, 32(t0) # store double value to memref of shape (1, 5)
    csrrci zero, 1984, 1 # SSR disable
    ret


.text
.globl kernel100_stride112
.p2align 2
kernel100_stride112:
    mv t2, a0
    mv t1, a1
    mv t0, a2
    li t3, 99
    scfgwi t3, 64 # dm 0 dim 0 bound
    li t3, 8
    scfgwi t3, 192 # dm 0 dim 0 stride
    li t3, 4
    scfgwi t3, 32 # dm 0 repeat
    li t3, 4
    scfgwi t3, 65 # dm 1 dim 0 bound
    li t3, 99
    scfgwi t3, 97 # dm 1 dim 1 bound
    li t3, 896
    scfgwi t3, 193 # dm 1 dim 0 stride
    li t3, -3192
    scfgwi t3, 225 # dm 1 dim 1 stride
    scfgwi zero, 33 # dm 1 repeat
    scfgwi t2, 768  # dm 0 dim 0 source
    scfgwi t1, 801  # dm 1 dim 1 source
    csrr x0, 0x7C2
    csrrsi zero, 1984, 1 # SSR enable
    mv t1, t0
    fld ft7, 0(t1) # load double from memref of shape (1, 5)
    fld ft6, 8(t0) # load double from memref of shape (1, 5)
    fld ft5, 16(t0) # load double from memref of shape (1, 5)
    fld ft4, 24(t0) # load double from memref of shape (1, 5)
    fld ft3, 32(t0) # load double from memref of shape (1, 5)
    li t1, 99
    frep.o t1, 5, 0, 0
    fmadd.d ft7, ft0, ft1, ft7
    fmadd.d ft6, ft0, ft1, ft6
    fmadd.d ft5, ft0, ft1, ft5
    fmadd.d ft4, ft0, ft1, ft4
    fmadd.d ft3, ft0, ft1, ft3
    mv t1, t0
    fsd ft7, 0(t1) # store double value to memref of shape (1, 5)
    fsd ft6, 8(t0) # store double value to memref of shape (1, 5)
    fsd ft5, 16(t0) # store double value to memref of shape (1, 5)
    fsd ft4, 24(t0) # store double value to memref of shape (1, 5)
    fsd ft3, 32(t0) # store double value to memref of shape (1, 5)
    csrrci zero, 1984, 1 # SSR disable
    ret
