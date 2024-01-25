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

static void intlui(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] = (s32) imm_utype;
}

static void intauipc(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] = (s32) imm_utype + riscv->pc.s64[0];
}

static void intaddi(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].s64[0] + simm;
}


static void intslti(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].s64[0] < (s64)simm ? 1 : 0;
}

static void intsltiu(riscv* riscv, u32 opcode) {
	unsigned _BitInt(12) uimm = imm_itype;
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] < (u64)uimm ? 1 : 0;
}

static void intxori(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] ^ simm;
}

static void intori(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] | simm;
}

static void intandi(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] & simm;
}

static void intslli(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] << shamt;
}

static void intsrli(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] >> shamt;
}

static void intsrai(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].s64[0] >> shamt;
}

static void intadd(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] + riscv->gpr[rs2].u64[0];
}

static void intsub(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] - riscv->gpr[rs2].u64[0];
}

static void intsll(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] << (riscv->gpr[rs2].u64[0] & 0x3f);
}

static void intslt(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].s64[0] < riscv->gpr[rs2].s64[0] ? 1 : 0;
}

static void intsltu(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] < riscv->gpr[rs2].u64[0] ? 1 : 0;
}

static void intxor(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] ^ riscv->gpr[rs2].u64[0];
}

static void intsrl(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] >> (riscv->gpr[rs2].u64[0] & 0x3f);
}

static void intsra(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].s64[0] >> (riscv->gpr[rs2].u64[0] & 0x3f);
}

static void intor(riscv* riscv, u32 opcode) {
	riscv->gpr[rd].s64[0] =  riscv->gpr[rs1].u64[0] | riscv->gpr[rs2].u64[0];
}

static void intlb(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].s64[0] =  (s8)read8(riscv, riscv->gpr[rs1].s64[0] + simm);
}

static void intlh(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].s64[0] =  (s16)read16(riscv, riscv->gpr[rs1].s64[0] + simm);
}

static void intlw(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].s64[0] =  (s32)read32(riscv, riscv->gpr[rs1].s64[0] + simm);
}

static void intlbu(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].u64[0] =  read8(riscv, riscv->gpr[rs1].s64[0] + simm);
}

static void intlhu(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].u64[0] =  read16(riscv, riscv->gpr[rs1].s64[0] + simm);
}

static void intlwu(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	riscv->gpr[rd].u64[0] =  read32(riscv, riscv->gpr[rs1].s64[0] + simm);
}

static void intsb(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_stype;
	write8(riscv, riscv->gpr[rs1].s64[0] + simm, riscv->gpr[rs2].u8[0]);
}

static void intsh(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_stype;
	write16(riscv, riscv->gpr[rs1].s64[0] + simm, riscv->gpr[rs2].u16[0]);
}

static void intsw(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_stype;
	write32(riscv, riscv->gpr[rs1].s64[0] + simm, riscv->gpr[rs2].u32[0]);
}

static void intjal(riscv* riscv, u32 opcode) {
	_BitInt(21) simm = imm_jtype;
	riscv->gpr[rd].u64[0] = riscv->pc.u64[0] +4;
	riscv->pc.s64[0] += simm;
}

static void intjalr(riscv* riscv, u32 opcode) {
	_BitInt(12) simm = imm_itype;
	auto temp = riscv->pc.u64[0] + 4;
	riscv->pc.u64[0] = (riscv->gpr[rs1].s64[0] + simm) & ~1;
	riscv->gpr[rd].u64[0] = temp;
}



static void intunimplemented(riscv* riscv, u32 opcode) {
	assert(false);
}



instrinfo instrs[] = {
	{"lui", 0b0110111, UTYPE},
	{"auipc", 0b0010111, UTYPE},
	{"jal", 0b1101111, JTYPE},
	{"jalr", 0b000000001100111, ITYPE},
	{"beq", 0b000000001100011, BTYPE},
	{"bne", 0b001000001100011, BTYPE},
	{"blt", 0b100000001100011, BTYPE},
	{"bge", 0b101000001100011, BTYPE},
	{"bltu", 0b110000001100011, BTYPE},
	{"bgeu", 0b111000001100011, BTYPE},
	{"lb", 0b000000000000011, ITYPE},
	{"lh", 0b001000000000011, ITYPE},
	{"lw", 0b010000000000011, ITYPE},
	{"lbu", 0b100000000000011, ITYPE},
	{"lhu", 0b101000000000011, ITYPE},
	{"sb", 0b000000000100011, STYPE},
	{"sh", 0b001000000100011, STYPE},
	{"sw", 0b010000000100011, STYPE},
	{"addi", 0b000000000010011, ITYPE},
	{"slti", 0b010000000010011, ITYPE},
	{"sltiu", 0b011000000010011, ITYPE},
	{"xori", 0b100000000010011, ITYPE},
	{"ori", 0b110000000010011, ITYPE},
	{"andi", 0b111000000010011, ITYPE},
	{"slli", 0b00000000000000000001000000010011, RTYPE},
	{"srli", 0b00000000000000000101000000010011, RTYPE},
	{"srai", 0b01000000000000000101000000010011, RTYPE},
	{"add",  0b00000000000000000000000000110011, RTYPE},
	{"sub",  0b01000000000000000000000000110011, RTYPE},
	{"sll",  0b00000000000000000000100000110011, RTYPE},
	{"slt",  0b00000000000000000001000000110011, RTYPE},
	{"sltu", 0b00000000000000000001100000110011, RTYPE},
	{"xor",  0b00000000000000000010000000110011, RTYPE},
	{"srl",  0b00000000000000000010100000110011, RTYPE},
	{"sra",  0b01000000000000000010100000110011, RTYPE},
	{"or",   0b00000000000000000011000000110011, RTYPE},
	{"and",  0b00000000000000000011100000110011, RTYPE},
	{"fence", 0b00000000000000000000000000001111, SPECIALTYPE},
	{"fence.tso", 0b10000011001100000000000000001111, SPECIALTYPE},
	{"pause", 0b00000001000000000000000000001111, SPECIALTYPE},
	{"ecall", 0b1110011, SPECIALTYPE},
	{"ebreak",0b00000000000100000000000001110011, SPECIALTYPE},
	{"lwu",   0b110000000000011, ITYPE},
	{"ld",    0b011000000000011, ITYPE},
	{"sd" ,   0b011000000100011, STYPE},
	{"slli" , 0b00000000000000000001000000010011, RTYPE},
	{"srli" , 0b00000000000000000101000000010011, RTYPE},
	{"srai" , 0b01000000000000000101000000010011, RTYPE},
	{"addiw", 0b00000000000000000000000000011011, ITYPE},
	{"slliw", 0b00000000000000000001000000011011, RTYPE},
	{"srliw", 0b00000000000000000101000000011011, RTYPE},
	{"sraiw", 0b01000000000000000101000000011011, RTYPE},
	{"addw",  0b00000000000000000000000000111011, RTYPE},
	{"subw",  0b01000000000000000000000000111011, RTYPE},
	{"sllw",  0b00000000000000000001000000111011, RTYPE},
	{"srlw",  0b00000000000000000101000000111011, RTYPE},
	{"sraw",  0b01000000000000000101000000111011, RTYPE},
  {"mul",   0b00000010000000000001000000110011, RTYPE},
	{"fence.i",  0b001000000001111, SPECIALTYPE},
  {"csrrw",   0b001000001110011, SPECIALTYPE},
  {"csrrs",   0b010000001110011, SPECIALTYPE},
  {"csrrc",   0b011000001110011, SPECIALTYPE},
  {"csrrwi",  0b101000001110011, SPECIALTYPE},
  {"csrrsi",  0b110000001110011, SPECIALTYPE},
  {"csrrci",  0b111000001110011, SPECIALTYPE},

  {"mul",    0b00000010000000000000000000110011, RTYPE},
  {"mulh",   0b00000010000000000001000000110011, RTYPE},
  {"mulhsu", 0b00000010000000000010000000110011, RTYPE},
	{"mulhu",  0b00000010000000000011000000110011, RTYPE},
	{"div",    0b00000010000000000100000000110011, RTYPE},
	{"divu",   0b00000010000000000101000000110011, RTYPE},
	{"rem",    0b00000010000000000110000000110011, RTYPE},
	{"remu",   0b00000010000000000111000000110011, RTYPE},

	{"mulw",    0b00000010000000000000000000111011, RTYPE},
	{"divw",    0b00000010000000000100000000111011, RTYPE},
	{"divuw",   0b00000010000000000101000000111011, RTYPE},
	{"remw",    0b00000010000000000110000000111011, RTYPE},
	{"remuw",   0b00000010000000000111000000111011, RTYPE},

  {"c.addi4spn", 0b0000000000000000, COMPRESSEDTYPE},
	{"c.fld", 0b0010000000000000, COMPRESSEDTYPE},
	{"c.lw", 0b0100000000000000, COMPRESSEDTYPE},
	{"c.flw", 0b0110000000000000, COMPRESSEDTYPE},
	{"c.ld", 0b0110000000000000, COMPRESSEDTYPE}, //todo: this and c.flw both have the same encoding. what do?
	{"c.fsd", 0b1010000000000000, COMPRESSEDTYPE},
	{"c.fsd", 0b1100000000000000, COMPRESSEDTYPE},
	{"c.fsd", 0b1110000000000000, COMPRESSEDTYPE},
	{"c.sd", 0b1110000000000000, COMPRESSEDTYPE}, // todo this and c.fsd have the same encoding
	//{"c.nop", 0b0000000000000001, COMPRESSEDTYPE}, // this is just c.addi
  {"c.addi", 0b0000000000000001, COMPRESSEDTYPE},
  {"c.jal", 0b0010000000000001, COMPRESSEDTYPE},
  {"c.addiw", 0b0010000000000001, COMPRESSEDTYPE}, //todo: same as c.jal but c.jal is for 32bit and addiw is 64 bit i think
  {"c.li", 0b0100000000000001, COMPRESSEDTYPE},
  {"c.addi16sp", 0b0110000000000001, COMPRESSEDTYPE},
  {"c.lui", 0b0110000000000001, COMPRESSEDTYPE}, //todo: same as c.addi16sp
  {"c.srli", 0b1000000000000001, COMPRESSEDTYPE},
  {"c.srai", 0b1000010000000001, COMPRESSEDTYPE},
  {"c.andi", 0b1000100000000001, COMPRESSEDTYPE},
  {"c.sub",  0b1000110000000001, COMPRESSEDTYPE},
  {"c.xor",  0b1000110000100001, COMPRESSEDTYPE},
  {"c.or",   0b1000110001000001, COMPRESSEDTYPE},
  {"c.and",  0b1000110001100001, COMPRESSEDTYPE},
  {"c.subw", 0b1001110000000001, COMPRESSEDTYPE},
  {"c.addw", 0b1001110000100001, COMPRESSEDTYPE},
  {"c.j",    0b1010000000000001, COMPRESSEDTYPE},
  {"c.beqz", 0b1100000000000001, COMPRESSEDTYPE},
  {"c.bnez", 0b1110000000000001, COMPRESSEDTYPE},
  {"c.slli", 0b0100000000000010, COMPRESSEDTYPE},
  {"c.fldsp",0b0010000000000010, COMPRESSEDTYPE},
  {"c.lwsp", 0b0010000000000010, COMPRESSEDTYPE}, //todo: same as c.fldsp
  {"c.flwsp",0b0110000000000010, COMPRESSEDTYPE},
  {"c.ldsp", 0b0110000000000010, COMPRESSEDTYPE}, //todo: same as c.flwsp
  {"c.jr",   0b1000000000000010, COMPRESSEDTYPE},
  {"c.mv",   0b1000000000000010, COMPRESSEDTYPE}, //todo: same as c.jr
  {"c.ebreak",0b1001000000000010, COMPRESSEDTYPE},
  {"c.jalr", 0b1001000000000010, COMPRESSEDTYPE}, //todo: same as c.ebreak
  {"c.add",  0b1001000000000010, COMPRESSEDTYPE}, //todo: same as c.ebreak and c.jalr
  {"c.fsdsp",0b1010000000000010, COMPRESSEDTYPE},
  {"c.swsp", 0b1100000000000010, COMPRESSEDTYPE},
  {"c.fswsp",0b1110000000000010, COMPRESSEDTYPE},
  {"c.sdsp", 0b1110000000000010, COMPRESSEDTYPE}, //todo: same as c.fswsp
};


void decode(riscv* riscv, u32 opcode) {

}