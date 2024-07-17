import os

def convert_dst_to_png(src_dir, dst_dir):
	"""
	Converts all .dst files in src_dir to .png files in dst_dir with the same name.

	Args:
		src_dir: Path to the directory containing .dst files.
		dst_dir: Path to the directory where .png files will be saved.
	"""
	i = 0
	for filename in os.listdir(src_dir):
		if filename.endswith(".dst"):
			i += 1
			filepath = os.path.join(src_dir, filename)
			# Read the DST file
			try:
				design = pyembroidery.read_dst(filepath)
			except Exception as e:
				print(f"Error reading {filepath}: {e}")
				continue

			# Create output filename with .png extension
			output_filename = os.path.splitext(filename)[0] + ".png"
			output_path = os.path.join(dst_dir, output_filename)

			# Convert design to PNG and save
			try:
				pyembroidery.write_png(design, output_path)
				print(f"Converted {i}/828")
			except Exception as e:
				print(f"Error converting {filepath}: {e}")

# Specify source and destination directories
source_dir = "data/datasets/dst/embroidery"
destination_dir = "data/datasets/dst/previews"

convert_dst_to_png(source_dir, destination_dir)
