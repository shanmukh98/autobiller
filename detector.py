# # Dependency Installation instructions
# %git clone https://github.com/shanmukh98/lang-segment-anything/tree/main
# %cd lang-segment-anything
# %pip install -e .


from PIL import Image
from lang_sam import LangSAM
from matplotlib import pyplot as plt
import numpy as np


'''
Sample Usage:

detector = ObjectDetector()

image_pil = Image.open("/home/shanmukh/experiments/autobiller/dataset/PXL_20241204_030419900.jpg").convert("RGB")   
resized_image = detector.resize(image_pil, 512)

text_prompt = "object, not hand, not background"
results = detector.predict(resized_image, text_prompt)
detector.plot_results(resized_image, results, text_prompt)

# remove hand from the image
hand = detector.predict(resized_image, "hand")
resized_image = detector.hide_mask(resized_image, hand)

cropped_image = detector.crop_image(resized_image, results)
detector.show_image(cropped_image)

'''

class ObjectDetector:
    def __init__(self):
        self.model = LangSAM(sam_type="sam2.1_hiera_tiny", gdino_type="tiny")
    
    def predict(self, image_pil, text_prompt):
        results = self.model.predict([image_pil], [text_prompt])
        return results
    
    def plot_results(self, image_pil, results, text_prompt):
        
        # Convert the mask to a numpy array
        mask = results[0]['masks'][0]

        # Plot the image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_pil)

        # Plot the bounding box
        box = results[0]['boxes'][0]
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor='red', facecolor='none', linewidth=2))

        # Plot the mask
        plt.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.5, cmap='jet')

        plt.title(f"Prediction for: {text_prompt}")
        plt.axis('off')
        plt.show()

    # given an image and results, hide the mask on the image by making the values zero within the mask
    def hide_mask(self, image_pil, results):
        mask = results[0]['masks'][0]
        image_np = np.array(image_pil)
        image_np[mask == 1] = 0
        return Image.fromarray(image_np)

    def crop_image(self, image_pil, results):
        box = results[0]['boxes'][0]
        cropped_image = image_pil.crop((box[0], box[1], box[2], box[3]))
        return cropped_image

    def show_image(self, image_pil):
        plt.imshow(image_pil)
        plt.axis('off')
        plt.show()

    def resize(self, image_pil, long_side):
        width, height = image_pil.size
        if width > height:
            new_width = long_side
            new_height = int(long_side * height / width)
        else:
            new_height = long_side
            new_width = int(long_side * width / height)
        return image_pil.resize((new_width, new_height))
    


