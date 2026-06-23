from PIL import Image
import os

codes = [
    (0,  0x0000000d5d628584),
    (1,  0x0000000d97f18b49),
    (2,  0x0000000dd280910e),
    (3,  0x0000000e479e9c98),
    (4,  0x0000000ebcbca822),
    (5,  0x0000000f31dab3ac),
    (6,  0x0000000056a5d085),
    (7,  0x000000010652e1d4),
    (8,  0x000000022b1dfead),
    (9,  0x0000000265ad0472),
    (10, 0x000000034fe91b86),
]

d = 6
border = 1
cell_size = 20
total = d + 2 * border

# Layout (cell coordinates, 0=top-left):
# - Black border: row/col 0, total-1
# - White border: row/col 1, total-2
# - Data: rows/cols 1..d

out_dir = os.path.join(os.environ['TEMP'], 'opencode')
os.makedirs(out_dir, exist_ok=True)

for tag_id, code in codes:
    img = Image.new('L', (total * cell_size, total * cell_size), 255)
    pixels = img.load()

    for row in range(total):
        for col in range(total):
            cx = col * cell_size
            cy = row * cell_size

            if row == 0 or row == total - 1 or col == 0 or col == total - 1:
                color = 0
            elif row == 1 or row == total - 2 or col == 1 or col == total - 2:
                color = 255
            else:
                data_row = row - border - 1
                data_col = col - border - 1
                bit_idx = (d - 1 - data_row) + d * data_col
                bit = (code >> bit_idx) & 1
                color = 255 if bit else 0

            for dy in range(cell_size):
                for dx in range(cell_size):
                    pixels[cx + dx, cy + dy] = color

    fp = os.path.join(out_dir, f'tag36h11_{tag_id}.png')
    img.save(fp)
    print(f'Saved: {fp} ({img.size[0]}x{img.size[1]})')

print('Done')
