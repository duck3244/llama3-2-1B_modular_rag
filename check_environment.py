#!/usr/bin/env python
"""
Python 3.10 버전 체크 및 환경 검증 스크립트
"""
import sys
import platform


def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    print(f"현재 Python 버전: {platform.python_version()}")
    
    if version.major == 3 and version.minor == 10:
        print("✓ Python 3.10이 올바르게 설치되어 있습니다.")
        return True
    else:
        print(f"✗ 오류: Python 3.10이 필요합니다. 현재 버전: {version.major}.{version.minor}")
        print("\n해결 방법:")
        print("1. Conda 사용:")
        print("   conda create -n rag-env python=3.10")
        print("   conda activate rag-env")
        print("\n2. pyenv 사용:")
        print("   pyenv install 3.10.13")
        print("   pyenv local 3.10.13")
        return False


def check_dependencies():
    """주요 패키지 설치 여부 확인"""
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'langchain': 'LangChain',
        'langgraph': 'LangGraph',
        'chromadb': 'ChromaDB',
    }
    
    print("\n패키지 확인:")
    all_installed = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} 설치됨")
        except ImportError:
            print(f"✗ {name} 미설치")
            all_installed = False
    
    if not all_installed:
        print("\n설치 방법:")
        print("pip install -r requirements.txt")
    
    return all_installed


def main():
    """메인 함수"""
    print("=" * 60)
    print("Python 3.10 환경 검증")
    print("=" * 60)
    
    version_ok = check_python_version()
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    if version_ok and deps_ok:
        print("환경 검증 완료! 프로젝트를 실행할 준비가 되었습니다.")
        print("\n실행 방법:")
        print("python llama_modular_rag/main.py")
    else:
        print("환경 설정이 필요합니다. 위의 지침을 따라주세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
