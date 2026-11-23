"""Genera el QR de la bibliografia/repositorio.  Uso: python generar_qr.py https://github.com/USUARIO/plenaria-teleportacion"""
import sys, qrcode
url = sys.argv[1]
img = qrcode.make(url, box_size=12, border=2)
img.save("QR_repositorio.png"); print("QR guardado en QR_repositorio.png ->", url)
