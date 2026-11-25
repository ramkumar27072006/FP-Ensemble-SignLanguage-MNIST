import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = docx.Document()

# Set standard margins (1 inch)
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Title
title_p = doc.add_paragraph()
title_run = title_p.add_run("Closed-Form Convolutional Learning via Ensembled Random Projection Targets")
title_run.font.name = "Times New Roman"
title_run.font.size = Pt(18)
title_run.font.bold = True
title_p.paragraph_format.space_after = Pt(24)

# Authors & Affiliations
author_p = doc.add_paragraph()
author_run = author_p.add_run("Ramkumar R\n")
author_run.font.name = "Times New Roman"
author_run.font.size = Pt(13)
author_run.font.bold = True

affil_run = author_p.add_run(
    "Department of Artificial Intelligence and Machine Learning\n"
    "Amrita Vishwa Vidyapeetham, Coimbatore, Tamil Nadu, India\n"
    "ORCID: 0009-0007-0076-8107\n"
)
affil_run.font.name = "Times New Roman"
affil_run.font.size = Pt(11)
author_p.paragraph_format.space_after = Pt(20)

# Corresponding Author Block
corr_p = doc.add_paragraph()
corr_header = corr_p.add_run("Corresponding Author:\n")
corr_header.font.name = "Times New Roman"
corr_header.font.size = Pt(11)
corr_header.font.bold = True

corr_text = corr_p.add_run(
    "Ramkumar R\n"
    "Department of Artificial Intelligence and Machine Learning\n"
    "Amrita Vishwa Vidyapeetham\n"
    "Amrita Nagar, Ettimadai, Coimbatore, Tamil Nadu 641112, India\n"
    "Institutional Email: cb.ai.u4aim24033@cb.students.amrita.edu\n"
    "Alternative Email: ramkumarashvanth09@gmail.com\n"
)
corr_text.font.name = "Times New Roman"
corr_text.font.size = Pt(10.5)
corr_p.paragraph_format.space_after = Pt(24)

# Acknowledgments
ack_p = doc.add_paragraph()
ack_header = ack_p.add_run("Acknowledgments:\n")
ack_header.font.name = "Times New Roman"
ack_header.font.size = Pt(11)
ack_header.font.bold = True

ack_text = ack_p.add_run(
    "The author expresses sincere gratitude to the Department of Artificial Intelligence and Machine Learning "
    "at Amrita Vishwa Vidyapeetham, Coimbatore, for providing the computational resources, facilities, and academic "
    "environment that supported this research."
)
ack_text.font.name = "Times New Roman"
ack_text.font.size = Pt(10.5)
ack_p.paragraph_format.space_after = Pt(18)

# Declarations / Disclosures
disc_p = doc.add_paragraph()
disc_header = disc_p.add_run("Declarations and Data Availability:\n")
disc_header.font.name = "Times New Roman"
disc_header.font.size = Pt(11)
disc_header.font.bold = True

disc_text = disc_p.add_run(
    "• Funding: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.\n"
    "• Conflict of Interest: The author declares no competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.\n"
    "• Code and Data Availability: All source code, trained models, and benchmark evaluation scripts supporting the findings of this study are openly available in the GitHub repository at: https://github.com/ramkumar27072006/FP-Ensemble-SignLanguage-MNIST."
)
disc_text.font.name = "Times New Roman"
disc_text.font.size = Pt(10)

out_path = r"C:\Users\R RAMKUMAR\.gemini\antigravity\scratch\FP-Ensemble-SignLanguage-MNIST\paper\Title_Page_Ramkumar.docx"
doc.save(out_path)
print("Title Page DOCX created successfully at:", out_path)
