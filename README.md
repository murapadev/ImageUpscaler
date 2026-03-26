# 🖼️ Image Upscaler

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.8+-blue)

> Image upscaler using Swin2SR model from Hugging Face Transformers.

---

## 📋 Tabla de contenidos

- [Características](#-características)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## ✨ Características

- �.scale **2x Upscaling**: Mejora resolución hasta 2x
- 🎮 **GPU Support**: Usa CUDA automáticamente
- 🧩 **Smart Tiling**: Maneja imágenes grandes sin OOM
- 🧵 **Seamless**: Une tiles sin artifacts

## 🛠️ Instalación

```bash
# Clonar
git clone https://github.com/murapadev/ImageUpscaler.git
cd ImageUpscaler

# Instalar
pip install -r requirements.txt
```

## 🚀 Uso

```bash
# Básico
python upscale.py input.jpg

# Con output específico
python upscale.py input.jpg --output result.jpg --scale 2
```

## 📝 Contribución

Las contribuciones son bienvenidas. Abre un issue o pull request.

## 📄 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.

---

*Hecho con ❤️ por [murapadev](https://github.com/murapadev)*
