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


void illegal(riscv* riscv){}

void decode(riscv* riscv) {
	u16 op16 = read16(riscv, pc64u);
	u32 op32 = read32(riscv, pc64u);

	switch(op16 & 0x3) {
		case 0: switch(op16 >> 13 & 0x7) {
			case 0: {u16 uimm = (op16 & 0x20 >> 2) | (op16 & 0x40 >> 4) | (op16 & 0x370 >> 1) | (op16 & 0x1800 >> 12); riscv->gpr[8+ (op16>>2&0x7)].u64[0] = riscv->gpr[2].u64[0] + uimm; }/*c.addi4spn*/ break;
			case 1: /*c.fld*/ illegal(riscv); break;
			case 2: {/*c.lw*/ u16 imm = (op16 & 0x20 << 1) | (op16 & 0x40 >> 4) | (op16 & 0x1c00 >> 7); riscv->gpr[8+(op16>>2&0x7)].s64[0] = (s32)read32(riscv,riscv->gpr[8+(op16>>7&0x7)].u64[0] + imm);}break;
			case 3: /*rv32c: c.flw OR rv64c: c.ld*/ break;
			case 5: /*c.fsd */ break;
			case 6: /*c.sw */ break;
			case 7: /*rv32c: c.fsw OR rv64c: c.sd*/ break;
		} break;
		case 1: switch(op16 >> 13 & 0x7) {
			case 0: /* c.nop if op16 == 1 else c.addi */ break;
			case 1: /* rv32c: c.jal OR rv64c: c.addiw*/ break;
			case 2: /*c.li*/ break;
			case 3: /*c.addi16sp OR c.lui*/ break;
			case 4: switch(op16 >> 10 & 0x3) {
				case 0: /*c.srli*/ break;
				case 1: /* c.srai*/ break;
				case 2: /* c.andi*/ break;
				case 3: switch(op16 >>5 & 0x3) {
					case 0: if(op16 >>12 & 1){/*c.subw*/} else{/*c.sub*/} break;
					case 1: if(op16 >>12 & 1){/*c.addw*/} else{/*c.xor*/} break;
					case 2: if(op16 >>12 & 1){/*invalid*/} else{/*c.or*/} break;
					default: illegal(riscv); break;
					} break;
				default: illegal(riscv); break;
				} break;
			case 5: /* c.j*/ break;
			case 6: /* c.beqz*/ break;
			case 7: /* c.bnez*/ break;
			} break;
		case 2: switch(op16 >> 13 & 0x7) {
			case 0: /*c.slli*/ break;
			case 1: /*c.fldsp*/ break;
			case 2: /*c.lwsp*/ break;
			case 3: /*c.ldsp*/ break;
			case 4: if((op16 >> 12 & 0x1) == 0) {
				if(op16>>2 & 0x1f) {
					/*c.jr*/
				}else{/*c.mv*/}
			}else {
				if((op16>>2 & 0x1f) == 0) {
					if((op16>>7&0x1f)==0) {
						/*c.ebreak*/
					}else {
						/*c.jalr*/
					}
				} else {/*c.add*/}
			} break;
			case 5: /*c.fsdsp*/ break;
			case 6: /*c.swsp */ break;
			case 7: /* c.fdsp*/ break;
		}
		case 3: {
			/* 32 bit opcode */
			switch(op32 >> 2 & 0x1f) {
				case 0b01101: /*lui */ break;
				case 0b00101: /*auipc */ break;
				case 0b11011: /*jal */ break;
				case 0b11001: if(op32>>12&0x7 == 0){/*jalr*/}else{/*???*/} break;
				case 0b11000: switch(op32>>12&0x7) {
					case 0: /*beq*/ break;
					case 1: /*bne*/ break;
					case 4: /*blt*/ break;
					case 5: /*bge*/ break;
					case 6: /*bltu*/ break;
					case 7: /*bgeu*/ break;
				} break;
				case 0b00000: switch(op32>>12&0x7) {
					case 0: /*lb*/ break;
					case 1: /*lh*/ break;
					case 2: /*lw*/ break;
					case 3: /*ld*/ break;
					case 4: /*lbu*/ break;
					case 5: /*lhu*/ break;
					case 6: /*lwu*/ break;
				} break;
				case 0b01000: switch(op32>>12&0x7) {
					case 0: /*sb*/ break;
					case 1: /*sh*/ break;
					case 2: /*sw*/ break;
					case 3: /*sd*/ break;
				} break;
				case 0b00100: switch(op32>>12&0x7) {
					case 0: /*addi*/ break;
					case 1: /*slli*/ break;
					case 2: /*slti*/ break;
					case 3: /*sltiu*/ break;
					case 4: /*xori*/ break;
					case 5: if(op32>>30&1){/*srai*/}else{/*srli*/} break;
					case 6: /*ori*/ break;
					case 7: /*andi*/ break;
				}break;
				case 0b01100: if(op32>>25&1) {
					switch(op32>>12&0x7) {
						case 0: /*mul*/ break;
						case 1: /*mulh*/ break;
						case 2: /*mulhsu*/ break;
						case 3: /*mulhu*/ break;
						case 4: /*div*/ break;
						case 5: /*divu*/ break;
						case 6: /*rem*/ break;
						case 7: /*remu*/ break;
					}
				}else {
					switch(op32>>12&0x7) {
						case 0: if(op32>>30&1){/*su*/}else{/*add*/} break;
						case 1: /*sll*/ break;
						case 2: /*slt*/ break;
						case 3: /*sltu*/ break;
						case 4: /*xor*/ break;
						case 5: if(op32>>30&1){/*sra*/}else{/*srl*/} break;
						case 6: /*or*/ break;
						case 7: /*and*/ break;
					}
				}break;
				case 0b00011: /*fence fence.tso pause fence.i*/ break;
				case 0b11100: /*ecall ebreak csrr* */ break;
				case 0b00110: switch(op32>>12&0x7) {
					case 0: /*addiw*/ break;
					case 1: /*slliw*/ break;
					case 5: if(op32>>30&1){/*sraiw*/}else{/*srliw*/} break;
				}break;
				case 0b01110: if(op32>>25&1) {
					switch(op32>>12&0x7) {
						case 0:/*mulw*/ break;
						case 4:/*divw*/ break;
						case 5:/*divuw*/ break;
						case 6:/*remw*/ break;
						case 7:/*remuw*/ break;
					}
				}else {
					switch(op32>>12&0x7) {
						case 0: if(op32>>30&1){/*subw*/}else{/*addw*/} break;
						case 1: /*sllw*/ break;
						case 5: if(op32>>30&1){/*sraw*/}else{/*srlw*/} break;
					}
				}
			}
		}
	}
}