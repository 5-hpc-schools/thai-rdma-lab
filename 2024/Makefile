# Makefile หลัก — สร้างทุกห้องปฏิบัติการ

LABS = lab01-verbs-basics lab02-ucx-messaging lab03-mpi-collectives \
       lab05-ucc-collectives lab06-competition-prep

# lab04-nccl-gpu ต้องมี CUDA จึงแยกออก
CUDA_LABS = lab04-nccl-gpu

.PHONY: all clean $(LABS) $(CUDA_LABS)

all: $(LABS)
	@echo "สร้างห้องปฏิบัติการทั้งหมดเสร็จสมบูรณ์ (ยกเว้น lab04 ที่ต้องมีซียูดีเอ)"

cuda: $(CUDA_LABS)

$(LABS):
	@echo "=== กำลังสร้าง $@ ==="
	$(MAKE) -C $@ || echo "[ข้าม] $@ — อาจต้องการไลบรารีเฉพาะ"

$(CUDA_LABS):
	@echo "=== กำลังสร้าง $@ (ต้องมีซียูดีเอ) ==="
	$(MAKE) -C $@ || echo "[ข้าม] $@ — ต้องมี CUDA Toolkit และ NCCL"

clean:
	@for lab in $(LABS) $(CUDA_LABS); do \
		echo "=== ทำความสะอาด $$lab ==="; \
		$(MAKE) -C $$lab clean 2>/dev/null || true; \
	done
