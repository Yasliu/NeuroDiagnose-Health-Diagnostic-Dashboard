import tensorflow as tf
import os

model_path = 'models/brain_tumor.h5'

print(f"📉 Processing {model_path}...")

# 1. Load the model with the custom logic
model = tf.keras.models.load_model(
    model_path, 
    custom_objects={'preprocess_input': tf.keras.applications.resnet50.preprocess_input}
)

# 2. Save WITHOUT Optimizer (Major savings)
model.save(model_path, include_optimizer=False)

# 3. Check Size
size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"✅ New Size: {size_mb:.2f} MB")

if size_mb > 100:
    print("⚠️ WARNING: Still too big for GitHub! We might need Git LFS.")
else:
    print("🎉 Success! Safe to push.")