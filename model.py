import re
import numpy as np
import pandas as pd

def clean_text(comment):
    comment = comment.strip().strip('"')
    comment = comment.replace('_', ' ')
    comment = comment.replace("\\\\", "\\")
    comment = comment.replace('\\n', ' ')
    comment = comment.replace('\\n', ' ')
    comment = comment.lower()
    comment = re.sub(r'@[A-Za-z0-9]+', '', comment)

    comment = re.sub('https?://[A-Za-z0-9./]+', '', comment)
    commment = re.sub("[^a-zA-Z]", " ", comment)
    comment = re.sub(r'[^\w\s]', '', comment)

    return comment


