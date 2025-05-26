"""
Diagnostic script for Psycore-V2.
This script helps identify and fix common issues with Psycore-V2.
"""

import os
import sys
import importlib
import traceback
import json

def check_environment():
    """Check the Python environment and libraries"""
    print("\n=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    required_modules = [
        "yaml", "boto3", "pinecone", "torch", "transformers", 
        "streamlit", "langchain", "openai", "anthropic"
    ]
    
    for module_name in required_modules:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ {module_name} is installed (version: {getattr(module, '__version__', 'unknown')})")
        except ImportError:
            print(f"✗ {module_name} is NOT installed")
        except Exception as e:
            print(f"? {module_name} check error: {str(e)}")

def check_credentials():
    """Check if necessary credentials are available"""
    print("\n=== Credentials Check ===")
    
    try:
        from src.system_manager import LocalCredentials
        print("✓ LocalCredentials module imported successfully")
        
        credential_names = [
            "AWS_IAM_KEY", "AWS_DEFAULT_REGION", 
            "S3_DOCUMENTS_BUCKET", "S3_TEXT_BUCKET", 
            "S3_IMAGES_BUCKET", "S3_GRAPHS_BUCKET",
            "PINECONE_INDEX", "PINECONE_API_KEY", "PINECONE_REGION"
        ]
        
        all_good = True
        for cred_name in credential_names:
            try:
                cred = LocalCredentials.get_credential(cred_name)
                if cred is not None:
                    print(f"✓ {cred_name} is set")
                else:
                    print(f"✗ {cred_name} is None")
                    all_good = False
            except Exception as e:
                print(f"✗ {cred_name} error: {str(e)}")
                all_good = False
        
        if all_good:
            print("\nAll credentials appear to be set properly.")
        else:
            print("\nSome credentials are missing or invalid.")
    
    except Exception as e:
        print(f"✗ Could not import LocalCredentials: {str(e)}")

def check_config():
    """Check the configuration file"""
    print("\n=== Configuration Check ===")
    
    try:
        import yaml
        
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        print("✓ config.yaml loaded successfully")
        
        sections = ["model", "graph_verification", "prompt_mode", "embedding", "logger", "document_range", "rag", "iteration"]
        
        for section in sections:
            if section in config:
                print(f"✓ {section} section is present")
            else:
                print(f"✗ {section} section is missing")
        
    except Exception as e:
        print(f"✗ Error checking config.yaml: {str(e)}")

def check_init_imports():
    """Try to import key modules to check for import errors"""
    print("\n=== Import Check ===")
    
    modules_to_check = [
        "src.system_manager.LoggerController",
        "src.system_manager.ConfigManager",
        "src.data.s3_handler.S3Handler",
        "src.kg.BERT_KG",
        "src.llm.ModelCatalogue",
        "src.vector_database.PineconeService",
        "src.main.IterativeStage"
    ]
    
    for module_path in modules_to_check:
        try:
            components = module_path.split(".")
            module_name = ".".join(components[:-1])
            class_name = components[-1]
            
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            print(f"✓ {module_path} imported successfully")
        except Exception as e:
            print(f"✗ {module_path} import error: {str(e)}")

def test_psycore_import():
    """Try to import and initialize Psycore to check for initialization errors"""
    print("\n=== Psycore Import Test ===")
    
    try:
        from psycore import Psycore
        print("✓ psycore module imported successfully")
        
        print("Attempting to initialize Psycore (this may take a moment)...")
        try:
            psycore = Psycore()
            print("✓ Psycore initialized successfully")
        except Exception as e:
            print(f"✗ Psycore initialization error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            
    except Exception as e:
        print(f"✗ Error importing psycore module: {str(e)}")

def check_s3_buckets():
    """Check if S3 buckets are accessible"""
    print("\n=== S3 Bucket Check ===")
    
    try:
        from src.data.s3_handler import S3Handler, S3Bucket
        from src.system_manager import LocalCredentials
        
        s3_creds = {
            "aws_iam": LocalCredentials.get_credential('AWS_IAM_KEY'),
            "region": LocalCredentials.get_credential('AWS_DEFAULT_REGION').secret_key,
            "buckets": {
                "documents": LocalCredentials.get_credential('S3_DOCUMENTS_BUCKET').secret_key,
                "text": LocalCredentials.get_credential('S3_TEXT_BUCKET').secret_key,
                "images": LocalCredentials.get_credential('S3_IMAGES_BUCKET').secret_key,
                "graphs": LocalCredentials.get_credential('S3_GRAPHS_BUCKET').secret_key
            }
        }
        
        s3_handler = S3Handler(s3_creds)
        print("✓ S3Handler initialized successfully")
        
        for bucket_type in [S3Bucket.DOCUMENTS, S3Bucket.TEXT, S3Bucket.IMAGES, S3Bucket.GRAPHS]:
            try:
                files = s3_handler.list_base_directory_files(bucket_type)
                print(f"✓ {bucket_type.name} bucket accessible ({len(files)} files found)")
            except Exception as e:
                print(f"✗ {bucket_type.name} bucket error: {str(e)}")
                
    except Exception as e:
        print(f"✗ S3 handler error: {str(e)}")

def suggest_fixes():
    """Suggest possible fixes for common issues"""
    print("\n=== Suggestions and Common Fixes ===")
    print("""
1. Virtual Environment Issues:
   - Make sure you're using the correct virtual environment
   - Try: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)
   - Install missing packages: pip install -r requirements.txt

2. Credential Issues:
   - Check if all AWS, Pinecone, and API credentials are properly set
   - Ensure .env file exists and has all required credentials

3. Document Issues:
   - Ensure there are documents in the S3 DOCUMENTS bucket
   - Check if documents have been preprocessed properly

4. Model Access Issues:
   - Verify you have API access to the LLMs specified in your config
   - Check if paid accounts and billing are set up correctly 

5. Running the Test:
   - Try running: python test_iterative_stage.py
   - Check logs for detailed error messages
   - Try with a simplified config with fewer documents
    """)

if __name__ == "__main__":
    print("=== Psycore-V2 Diagnostic Tool ===")
    print("Running diagnostic checks to identify potential issues...")
    
    try:
        check_environment()
        check_credentials()
        check_config()
        check_init_imports()
        test_psycore_import()
        check_s3_buckets()
        suggest_fixes()
        
        print("\n=== Diagnostic Complete ===")
        print("Use the information above to identify and fix issues with your Psycore-V2 setup.")
    except Exception as e:
        print(f"\nError during diagnostics: {str(e)}")
        traceback.print_exc()
