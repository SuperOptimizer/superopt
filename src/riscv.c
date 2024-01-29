#include "common.h"

typedef struct riscv {
	r64 gpr[32];
	r64 pc;
	u8* mem;
} riscv;


typedef enum instrenum {
	RTYPE = 0,
	ITYPE,
	STYPE,
	BTYPE,
	UTYPE,
	JTYPE,
	SPECIALTYPE,
	COMPRESSEDTYPE,
} instrenum;

typedef void (*intfunc)(riscv* riscv, u32 opcode);

typedef struct instrinfo {
	char* instr;
	u32 mask;
	instrenum type;
	intfunc intfunc;
} instrinfo;

u8 read8(riscv* riscv, u64 addr) {
	return riscv->mem[addr];
}

u16 read16(riscv* riscv, u64 addr) {
	return *(u16*)&riscv->mem[addr];
}

u32 read32(riscv* riscv, u64 addr) {
	return *(u32*)&riscv->mem[addr];
}

u64 read164(riscv* riscv, u64 addr) {
	return *(u64*)&riscv->mem[addr];
}

void write8(riscv* riscv, u64 addr, u8 data) {
	riscv->mem[addr] = data;
}

void write16(riscv* riscv, u64 addr, u16 data) {
	*(u16*)&riscv->mem[addr] = data;
}

void write32(riscv* riscv, u64 addr, u32 data) {
	*(u32*)&riscv->mem[addr] = data;
}

void write64(riscv* riscv, u64 addr, u64 data) {
	*(u64*)&riscv->mem[addr] = data;
}


#define rd (opcode >> 7 & 0x1f)
#define rs1 (opcode >> 15 & 0x1f)
#define rs2 (opcode >> 20 & 0x1f)
#define shamt (opcode >> 20 & 0x3f) // todo: handle 32 vs 64 bit
#define imm_itype (opcode >> 20)
#define imm_stype ((opcode >> 7 & 0x1f) | ((opcode >> 25) << 5))
#define imm_btype ((opcode >> 7 & 0x1e) | ((opcode >> 25 & 0x3f) << 5) | ((opcode >> 7 & 1) <<11) | ((opcode >> 31 & 1) <<12))
#define imm_utype (opcode & 0xfffff000)
#define imm_jtype (((opcode >> 31 & 1) << 20) | ((opcode >> 21 & 0x3ff) << 1) | ((opcode >> 20 & 1) << 11) | ((opcode >> 12 & 0xff) << 12))

#define pc64s  riscv->pc.s64[0]
#define pc64u  riscv->pc.u64[0]
#define rd64s  riscv->gpr[rd].s64[0]
#define rd64u  riscv->gpr[rd].u64[0]
#define rs164s riscv->gpr[rs1].s64[0]
#define rs164u riscv->gpr[rs1].u64[0]
#define rs264s riscv->gpr[rs2].s64[0]
#define rs264u riscv->gpr[rs2].u64[0]

#define rd32s  riscv->gpr[rd].s32[0]
#define rd32u  riscv->gpr[rd].u32[0]
#define rs132s riscv->gpr[rs1].s32[0]
#define rs132u riscv->gpr[rs1].u32[0]
#define rs232s riscv->gpr[rs2].s32[0]
#define rs232u riscv->gpr[rs2].u32[0]

void decode(riscv* riscv) {
	u16 op16 = read16(riscv, pc64u);
	u32 op32 = read32(riscv, pc64u);

	switch(op16 & 0x3) {
		case 0: {
			switch(op16 >> 13 & 0x7) {
				case 0: /*c.addi4spn*/ break;
				case 1: /*c.fld*/ break;
				case 2: /*c.lw*/ break;
				case 3: /*c.flw OR c.ld*/ break;
				case 5: /*c.fsd */ break;
				case 6: /*c.sw */ break;
				case 7: /*c.fsw OR c.sd*/ break;
			}break;
		} break;
		case 1: {
			switch(op16 >> 13 & 0x7) {
				case 0: /* c.nop if op16 == 1 else c.addi */ break;
				case 1: /*c.jal OR c.addiw*/ break;
				case 2: /*c.li*/ break;
				case 3: /*c.addi16sp OR c.lui*/ break;
				case 4:
				case 5: /*c.fsd */ break;
				case 6: /*c.sw */ break;
				case 7: /*c.fsw OR c.sd*/ break;
			}break;
		}
	}
}