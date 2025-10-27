#!/bin/bash
# OSMesa渲染器安装脚本 (WSL2)

echo "=========================================="
echo "🔧 安装OSMesa渲染支持"
echo "=========================================="
echo ""

# 检查是否在WSL2中
if grep -q microsoft /proc/version; then
    echo "✓ 检测到WSL2环境"
else
    echo "⚠️  警告: 不在WSL2环境中"
fi

# 安装OSMesa库
echo ""
echo "[1/4] 安装OSMesa库..."
sudo apt-get update
sudo apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    patchelf

if [ $? -eq 0 ]; then
    echo "  ✓ OSMesa库安装成功"
else
    echo "  ✗ 安装失败"
    exit 1
fi

# 安装Python OpenGL支持
echo ""
echo "[2/4] 安装Python OpenGL支持..."
pip install PyOpenGL PyOpenGL_accelerate

if [ $? -eq 0 ]; then
    echo "  ✓ PyOpenGL安装成功"
else
    echo "  ✗ 安装失败"
    exit 1
fi

# 重新安装mujoco-py (如果需要)
echo ""
echo "[3/4] 检查mujoco-py..."
python -c "import mujoco_py" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  ℹ️  mujoco-py未安装或需要重新编译"
    pip uninstall -y mujoco-py
    pip install mujoco-py
fi

# 验证安装
echo ""
echo "[4/4] 验证OSMesa安装..."

python3 << 'EOF'
import os
os.environ['MUJOCO_GL'] = 'osmesa'

try:
    import mujoco_py
    print("  ✓ mujoco-py可以使用OSMesa")
except Exception as e:
    print(f"  ⚠️  mujoco-py测试: {e}")

try:
    from OpenGL import GL
    print("  ✓ PyOpenGL可用")
except Exception as e:
    print(f"  ⚠️  PyOpenGL测试: {e}")
EOF

echo ""
echo "=========================================="
echo "✅ OSMesa安装完成！"
echo "=========================================="
echo ""
echo "📝 使用方法："
echo "  export MUJOCO_GL=osmesa"
echo "  export PYOPENGL_PLATFORM=osmesa"
echo "  python 05-smolvla_use_libero_fixed.py"
echo ""