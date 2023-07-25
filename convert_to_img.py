from pdf2image import convert_from_path
pages = convert_from_path('./Leifson.pdf', 500)
for count, page in enumerate(pages):
    page.save(f'out{count}.jpg', 'JPEG')