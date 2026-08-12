import pypdf
from pathlib import Path

BASE_DIR = Path("c:/Users/User/Desktop/best_resume_maker")
resume_path = BASE_DIR / "resume.pdf"

reader = pypdf.PdfReader(str(resume_path))
page = reader.pages[0]
contents = page.get_contents()
data = contents.get_data()

# Find Skills text positioning in the stream
idx = data.find(b"Languages/Data:")
if idx != -1:
    print("Found Skills section stream snippet:")
    # Print 1500 bytes around this index to understand how lines are arranged
    print(data[idx-100:idx+1400].decode('latin-1', errors='ignore'))
else:
    print("Languages/Data: not found")
