import subprocess


def combine(raw_folder, output_folder):
    subprocess.run(["sudo cd /Volumes/Elements 1"])
    subprocess.run([f"sudo mkdir {output_folder}"])
    subprocess.run(["sudo", "find", raw_folder,  ""])
    subprocess.run([f"sudo cd {output_folder}"])
    subprocess.run(["find . -type f ! -iname -type d \( -name backups -o -name wiki -o -name music \) -prune -false -o -type f -exec cp -nv '{}' music/ \;\"*.mp3\" -delete"])

if __name__ == "__main__":
    combine("music_raw", "music")
