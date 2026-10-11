from pathlib import Path
print('\n'.join(str(p) for p in sorted(Path('runs/main_r16_lr1e4').glob('checkpoint-*'))))
