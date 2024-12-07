import os
from pathlib import Path
import PyPDF2
from docx import Document
from langdetect import detect, detect_langs
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file."""
    try:
        doc = Document(docx_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if pdf_reader.is_encrypted:
                print(f"PDF is encrypted: {pdf_path}")
                return None
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def detect_document_languages(text):
    """Detect languages in the document with confidence scores."""
    try:
        # Get detailed language probabilities
        languages = detect_langs(text)
        return [(lang.lang, lang.prob) for lang in languages]
    except:
        # Fallback to simple detection
        try:
            return [(detect(text), 1.0)]
        except:
            return [('unknown', 0.0)]

def detect_document_type_with_gemini(text):
    """Detect document type using Gemini AI."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Create a prompt that asks Gemini to identify the document type
        prompt = """Please analyze this legal document and identify its specific type.
        Focus on determining if it's one of these types:
        - NDA (Non-Disclosure Agreement)
        - Contract
        - Agreement (specify type if possible)
        - Power of Attorney
        - Loan Agreement
        - Share Purchase Agreement
        - Partnership Agreement
        - Service Agreement
        - Employment Agreement
        - Other (please specify)

        Provide your response in this format:
        Type: [document type]
        Confidence: [HIGH/MEDIUM/LOW]
        Reasoning: [brief explanation]

        First 1000 characters of the document:
        """
        
        # Use first 1000 characters for type detection
        sample_text = text[:1000] + "..." if len(text) > 1000 else text
        full_prompt = f"{prompt}\n{sample_text}"
        
        response = model.generate_content(full_prompt)
        response_text = response.text
        
        # Parse the response to extract the document type
        type_line = [line for line in response_text.split('\n') if line.startswith('Type:')]
        if type_line:
            doc_type = type_line[0].replace('Type:', '').strip().lower()
            # Normalize document types
            if 'nda' in doc_type or 'non-disclosure' in doc_type:
                return 'nda'
            elif 'agreement' in doc_type:
                return 'agreement'
            elif 'contract' in doc_type:
                return 'contract'
            else:
                return doc_type
        return 'default'
    except Exception as e:
        print(f"Error detecting document type: {str(e)}")
        return 'default'

def get_summarization_prompt(doc_type, language):
    """Get document-type specific summarization prompt."""
    prompts = {
        'nda': {
            'en': """Please analyze this Non-Disclosure Agreement and provide:
                    1. Key parties involved
                    2. Scope of confidentiality
                    3. Duration of the agreement
                    4. Key obligations and restrictions
                    5. Notable exceptions or special clauses""",
            'id': """Mohon analisis Perjanjian Kerahasiaan ini dan berikan:
                    1. Pihak-pihak utama yang terlibat
                    2. Ruang lingkup kerahasiaan
                    3. Durasi perjanjian
                    4. Kewajiban dan pembatasan utama
                    5. Pengecualian atau klausul khusus yang penting"""
        },
        'contract': {
            'en': """Please analyze this Contract and provide:
                    1. Type of contract and main purpose
                    2. Key parties and their roles
                    3. Main terms and conditions
                    4. Key obligations of each party
                    5. Important dates and deadlines
                    6. Notable special provisions""",
            'id': """Mohon analisis Kontrak ini dan berikan:
                    1. Jenis kontrak dan tujuan utama
                    2. Pihak-pihak utama dan peran mereka
                    3. Syarat dan ketentuan utama
                    4. Kewajiban utama setiap pihak
                    5. Tanggal dan tenggat waktu penting
                    6. Ketentuan khusus yang penting"""
        },
        'agreement': {
            'en': """Please analyze this Agreement and provide:
                    1. Type and purpose of the agreement
                    2. Parties involved and their roles
                    3. Key terms and conditions
                    4. Rights and obligations
                    5. Duration and termination conditions
                    6. Special provisions or notable clauses""",
            'id': """Mohon analisis Perjanjian ini dan berikan:
                    1. Jenis dan tujuan perjanjian
                    2. Pihak-pihak yang terlibat dan peran mereka
                    3. Syarat dan ketentuan utama
                    4. Hak dan kewajiban
                    5. Durasi dan kondisi pengakhiran
                    6. Ketentuan khusus atau klausul penting"""
        },
        'default': {
            'en': """Please analyze this legal document and provide:
                    1. Document type and purpose
                    2. Key parties involved
                    3. Main provisions and terms
                    4. Important dates or deadlines
                    5. Notable special clauses or conditions""",
            'id': """Mohon analisis dokumen hukum ini dan berikan:
                    1. Jenis dan tujuan dokumen
                    2. Pihak-pihak utama yang terlibat
                    3. Ketentuan dan syarat utama
                    4. Tanggal atau tenggat waktu penting
                    5. Klausul atau kondisi khusus yang penting"""
        }
    }
    
    # Default to English if language not supported
    lang = 'en' if language not in ['en', 'id'] else language
    return prompts.get(doc_type, prompts['default'])[lang]

def analyze_with_gemini(text, doc_type, language):
    """Analyze document using Gemini AI with type-specific prompts."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = get_summarization_prompt(doc_type, language)
        
        # Truncate text if too long (Gemini has token limits)
        max_chars = 30000  # Adjust based on Gemini's actual limits
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[Document truncated due to length...]"
        
        full_prompt = f"{prompt}\n\nDocument text:\n{text}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error using Gemini API: {str(e)}"

def process_legal_documents(base_path):
    """Process all legal documents in the given directory and its subdirectories."""
    results = []
    base_path = Path(base_path)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.pdf', '.docx')) and not file.startswith('~$'):  # Exclude temp files
                file_path = Path(root) / file
                print(f"\nProcessing: {file_path}")
                
                # Extract text based on file type
                if file.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                else:  # .docx
                    text = extract_text_from_docx(file_path)
                
                if text:
                    # Detect language
                    languages = detect_document_languages(text)
                    primary_language = languages[0][0] if languages else 'unknown'
                    
                    # Use Gemini to detect document type
                    print("Detecting document type with Gemini...")
                    doc_type = detect_document_type_with_gemini(text)
                    print(f"Detected document type: {doc_type}")
                    
                    # Analyze with Gemini
                    print(f"Analyzing document (type: {doc_type}, language: {primary_language})")
                    analysis = analyze_with_gemini(text, doc_type, primary_language)
                    
                    results.append({
                        'file_path': str(file_path),
                        'doc_type': doc_type,
                        'languages': languages,
                        'analysis': analysis
                    })
                    
                    print(f"Analysis complete for: {file}\n")
                    print("-" * 80)
                    print(f"Analysis results:\n{analysis}")
                    print("-" * 80)
    
    return results 