import os, glob
import sys

from pydub import AudioSegment
import subprocess
import shutil

if __name__ == "__main__":
    shutil.rmtree('temp', ignore_errors=True)
    os.makedirs('temp', exist_ok=True)
    files = glob.glob('input/*.m4a')
    for index, file in enumerate(files):
        full_name = os.path.basename(file)
        name_without_ext = os.path.splitext(full_name)[0]
        subprocess.run(
            f"ffmpeg -i input/{full_name} temp/{name_without_ext}.wav 2> /dev/null",
            shell=True,
            executable="/bin/zsh"
        )
        sys.stdout.write(f'\rConverting to .wav {index+1}/{len(files)}')

    print('')

    files = glob.glob('temp/*.wav')
    shutil.rmtree('trimmed', ignore_errors=True)
    os.makedirs('trimmed', exist_ok=True)
    for index, file in enumerate(files):
        full_name = os.path.basename(file)
        subprocess.run(
            f"python3 trim_silence.py --input temp/{full_name} --output trimmed/{full_name} --silence-dur=0.1",
            shell=True,
            executable="/bin/zsh"
        )
        sys.stdout.write(f'\rTrimming {index+1}/{len(files)}')

    print("")

    files = glob.glob('trimmed/*.wav')
    shutil.rmtree('output', ignore_errors=True)
    os.makedirs('output', exist_ok=True)
    for index, file in enumerate(files):
        full_name = os.path.basename(file)
        name_without_ext = os.path.splitext(full_name)[0]
        AudioSegment.from_wav(f"trimmed/{full_name}").export(
            f"output/{name_without_ext}.mp3",
            format="mp3",
            codec='mp3',
            bitrate="320"
        )
        sys.stdout.write(f'\rExporting as .mp3 {index+1}/{len(files)}')

    shutil.rmtree('trimmed', ignore_errors=True)
    shutil.rmtree('temp', ignore_errors=True)
    print(f"\nSuccessfully trimmed {len(files)} files!")

