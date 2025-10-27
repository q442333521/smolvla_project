#!/bin/bash
echo "=== Testing LIBERO Installation ==="
echo ""

# 设置环境
export PATH="/root/anaconda3/envs/libero/bin:$PATH"

echo "1. Testing Python import..."
python << 'ENDPY'
try:
    import libero
    print("   ✅ LIBERO can be imported")
except Exception as e:
    print(f"   ❌ Failed: {e}")
ENDPY

echo ""
echo "2. Testing benchmark module..."
python << 'ENDPY'
try:
    from libero.libero import benchmark
    print("   ✅ benchmark module works")
    bench = benchmark.get_benchmark("libero_spatial")
    print(f"   ✅ libero_spatial benchmark created")
except Exception as e:
    print(f"   ❌ Failed: {e}")
ENDPY

echo ""
echo "3. Checking installed packages..."
pip list | grep -E "(libero|robosuite|gym)" || echo "   No packages found"

echo ""
echo "=== Test Complete ==="
