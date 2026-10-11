"""Extrai ZIPs oficiais, inclusive arquivos ZIP internos, sem escapar do destino."""
import argparse
import zipfile
from pathlib import Path


def unpack(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Caminho inesperado no ZIP: " + item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Link simbolico inesperado no ZIP.")
        z.extractall(destination)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("destination")
    a = ap.parse_args()
    root = Path(a.destination)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use um diretorio de extracao vazio.")
    unpack(Path(a.archive), root)
    seen = set()
    while True:
        todo = [p for p in root.rglob("*.zip") if str(p.resolve()) not in seen]
        if not todo:
            break
        for p in todo:
            print("Extraindo arquivo interno:", p)
            seen.add(str(p.resolve()))
            unpack(p, p.parent)
    print("Bancos SQLite encontrados:", len(list(root.rglob("*.sqlite"))))
