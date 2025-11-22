"""
Script de verificação de GPU antes de executar o pipeline.
"""
import torch

print("="*70)
print("🔍 VERIFICAÇÃO DE GPU")
print("="*70)

print(f"\n📦 PyTorch version: {torch.__version__}")
print(f"🖥️  CUDA disponível: {'✅ SIM' if torch.cuda.is_available() else '❌ NÃO'}")

if torch.cuda.is_available():
    print(f"📊 Número de GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"      Nome: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      Memória total: {props.total_memory / 1e9:.2f} GB")
        print(f"      Compute capability: {props.major}.{props.minor}")
        print(f"      Multi-processors: {props.multi_processor_count}")
    
    # Testar alocação
    print(f"\n🧪 Teste de alocação GPU...")
    try:
        x = torch.randn(1000, 1000).cuda()
        print(f"   ✅ Tensor alocado na GPU com sucesso!")
        print(f"   Device: {x.device}")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ❌ Erro ao alocar: {e}")
else:
    print("\n⚠️  GPU não disponível. Pipeline rodará em CPU (mais lento).")
    print("   Para habilitar GPU, verifique:")
    print("   1. Driver NVIDIA instalado")
    print("   2. CUDA Toolkit instalado")
    print("   3. PyTorch com suporte CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*70)
