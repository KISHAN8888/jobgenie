import docx
import google.generativeai as genai

genai.configure(api_key="AIzaSyBgXXTa043ZpKYxIYfGOQ6P91fBD6Bkfxk")
# The Gemini 1.5 models are versatile and work with both text-only and multimodal prompts
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to extract text from DOCX file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to get skills and suitable roles from LLM
def get_skills_and_roles(resume_text):
    # Load the pre-trained language model pipeline for text generation
    skillstext ="This is a resume. Give all the skills of the candidate seperated with commas."+ resume_text
    rolestext= "This is a resume, follow this format strictly. format : Give all the role names only to which candidate can apply seperated with commas." +resume_text
    skills = model.generate_content(skillstext).text
    roles = model.generate_content(rolestext).text
    return skills, roles

# Main function
def main():
    # Path to the DOCX resume file
    file_path="C:\\Users\\kisha\\Documents\\jobease\\Kishan Tripathi Resume.docx"

    # Extract text from resume
    resume_text = extract_text_from_docx(file_path)
    print("Resume Text Extracted:\n", resume_text)

    # Get skills and suitable roles using LLM
    skills, roles = get_skills_and_roles(resume_text)
    print("\nExtracted Skills:\n", skills)
    print("\nSuitable Roles:\n", roles)

if __name__ == "__main__":
    main()
