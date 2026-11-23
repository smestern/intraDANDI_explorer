jupyter nbconvert --to html --TagRemovePreprocessor.remove_input_tags='{"hide_code"}' dandi_scraper/notes_on_intra_ephys.ipynb
# self-contained markdown + images for the blog
python -I tools/export_blog.py
