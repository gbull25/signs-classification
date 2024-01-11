
import glob
import random

path = "services/tg_bot/sample_images"
filename = random.choice(glob.glob(path))

print(len(glob.glob(path)))