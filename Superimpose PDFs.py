from pdf2image import convert_from_path
from PIL import Image, ImageChops
import os

def overlay_pdfs(base_pdf_path, overlay_pdf_path, output_pdf_path, blend_method="blend", blend_alpha=0.5):
    # Convert each page of both PDFs to images
    base_images = convert_from_path(base_pdf_path)
    overlay_images = convert_from_path(overlay_pdf_path)

    assert len(base_images) == len(overlay_images), "PDFs must have same number of pages"

    blended_images = []

    for base_img, overlay_img in zip(base_images, overlay_images):
        # Ensure same size
        if base_img.size != overlay_img.size:
            overlay_img = overlay_img.resize(base_img.size)

        if blend_method == "blend":
            blended = Image.blend(base_img.convert("RGBA"), overlay_img.convert("RGBA"), alpha=blend_alpha)
        elif blend_method == "add":
            blended = ImageChops.add(base_img.convert("RGB"), overlay_img.convert("RGB"), scale=2.0)
        elif blend_method == "subtract":
            blended = ImageChops.subtract(base_img.convert("RGB"), overlay_img.convert("RGB"))
        else:
            raise ValueError("Unknown blend method")

        blended_images.append(blended.convert("RGB"))  # Must be RGB to save as PDF

    # Save all blended pages as a new PDF
    blended_images[0].save(output_pdf_path, save_all=True, append_images=blended_images[1:])


overlay_pdfs(
    base_pdf_path="/Users/summer/Downloads/dolphins-master/code/Kidney Main Code/results/Kidney_z/kidneybase.pdf",
    overlay_pdf_path="/Users/summer/Downloads/dolphins-master/code/Kidney Main Code/Volumes/Health Files/censoring_results/8_day_spacing_kidney.pdf",
    output_pdf_path="/Users/summer/Downloads/dolphins-master/code/Kidney Main Code/results/Kidney_z/kidney_8_day_super.pdf"
)


