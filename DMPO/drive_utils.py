import os
import io
import shutil
import socket
import ssl
from transformers import TrainerCallback
from transformers.trainer import logger
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import time


SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_drive():
    """
    Authenticates and authorizes the application, returning credentials.
    Run this file before running the training script to ensure token.json generated.
    You need to have credentials.json file in the same directory.
    See https://console.cloud.google.com/apis/credentials?project=innate-marking-470300-p8.
    """
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', 
                scopes=SCOPES, 
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            print("Please go to this URL and authorize the app:")
            print(auth_url)
            
            code = input("Enter the authorization code from the URL: ")
            
            flow.fetch_token(code=code)
            creds = flow.credentials
            
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

creds = authenticate_google_drive()


class GoogleDriveUploaderCallback(TrainerCallback):
    """
    A self-defined TrainerCallback for uploading checkpoints to Google Drive.
    """
    def __init__(self, config):
        self.drive_service = build('drive', 'v3', credentials=creds)
        self.parent_id = config.google_drive_parent_folder_id
        self.is_main_process = config.process_index == 0
        if self.is_main_process:
            self.target_folder_id = self._get_or_create_folder(config.run_name, self.parent_id)

    def _get_or_create_folder(self, folder_name, parent_id=None):
        """
        Search for a folder in Drive or create it if it doesn't exist, returning its ID.
        If parent_id is None, search in the root directory.
        """
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        else:
            query += " and 'root' in parents"

        max_retries = 10; max_delay = 30
        for attempt in range(max_retries):
            try:
                results = self.drive_service.files().list(q=query, spaces='drive', fields='files(id)').execute()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} to query folder '{folder_name}' failed with error:\n"
                                   f"{e}\n"
                                   f"Retrying in {min(max_delay, 2 ** attempt)} seconds...")
                    time.sleep(min(max_delay, 2 ** attempt))
                else:
                    logger.error(f"All {max_retries} attempts failed for folder '{folder_name}'. Error:\n{e}")
                    raise

        existing_folders = results.get('files', [])
        if existing_folders:
            logger.info(f"Found existing folder for run '{folder_name}'.")
            return existing_folders[0]['id']
        else:
            logger.info(f"Creating new folder for run '{folder_name}'.")
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            for attempt in range(max_retries):
                try:
                    folder = self.drive_service.files().create(body=file_metadata, fields='id').execute()
                    return folder.get('id')
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed with error:\n"
                                   f"{e}\n"
                                   f"Retrying in {min(max_delay, 2 ** attempt)} seconds...")
                    time.sleep(min(max_delay, 2 ** attempt))
            logger.error(f"All {max_retries} attempts failed for folder '{folder_name}'. Error:\n{e}")
            raise

    def upload_folder(self, folder_path, parent_id=None):
        """
        Recursively uploads a local folder to Google Drive.
        Return 0 if all files uploaded successfully, otherwise return 1.
        """
        folder_name = os.path.basename(folder_path)
        
        # Check if the folder already exists to avoid duplication
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                results = self.drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} to locate folder '{folder_name}' failed with error:\n"
                                   f"{e}\n"
                                   "Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    logger.error(f"All {max_retries} attempts failed for folder '{folder_name}'. Error:\n{e}")
                    return 1

        existing_folders = results.get('files', [])
        if existing_folders:
            folder_id = existing_folders[0]['id']
            logger.info(f"Folder '{folder_name}' already exists. Uploading content inside.")
        else:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id:
                file_metadata['parents'] = [parent_id]
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    folder = self.drive_service.files().create(body=file_metadata, fields='id').execute()
                    folder_id = folder.get('id')
                    logger.info(f"Created folder '{folder_name}' with ID: {folder_id}")
                    break
                except Exception as e:
                    logger.error(f"Failed to create folder '{folder_name}' (attempt {attempt + 1}/{max_retries}):\n{e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying folder creation for '{folder_name}' in 5 seconds...")
                        time.sleep(5)
                    else:
                        logger.error(f"All {max_retries} attempts failed for folder '{folder_name}'. Error:\n{e}")
                        return 1

        exit_code = 0
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            try:
                if os.path.isdir(item_path):
                    exit_code += self.upload_folder(item_path, folder_id)
                else:
                    exit_code += self.upload_file(item_path, folder_id)
            except Exception as e:
                logger.error(f"Exception occurred while uploading '{item_path}': {e}")
                exit_code += 1

        return exit_code > 0

    def upload_file(self, file_path, parent_id):
        """
        Uploads a single file to a Google Drive folder.
        Return 0 if uploaded successfully, otherwise return 1.
        """
        file_name = os.path.basename(file_path)
        file_metadata = {
            'name': file_name,
            'parents': [parent_id]
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                media = MediaIoBaseUpload(io.FileIO(file_path, 'rb'),
                                         mimetype='application/octet-stream',
                                         resumable=True)
                logger.info(f"Uploading file '{file_name}' (attempt {attempt+1}/{max_retries})...")
                self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                logger.info(f"File '{file_name}' uploaded successfully.")
                return 0
            except (ssl.SSLEOFError, socket.error) as ssl_e:
                logger.warning(f"SSL/network error while uploading file '{file_name}': {ssl_e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying upload for '{file_name}' in 5 seconds...")
                    time.sleep(5)
                else:
                    logger.error(f"Failed to upload file '{file_name}' after {max_retries} attempts due to SSL/network error.")
                    return 1
            except Exception as e:
                logger.warning(f"Failed to upload file '{file_name}' (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying upload for '{file_name}' in 5 seconds...")
                    time.sleep(5)
                else:
                    logger.error(f"Failed to upload file '{file_name}' after {max_retries} attempts.")
                    return 1

    def on_save(self, args, state, control, **kwargs):
        """
        Called after a checkpoint is saved.
        """
        if self.is_main_process:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            logger.info(f"Saving checkpoint to Google Drive: {checkpoint_dir}")
            exit_code = self.upload_folder(checkpoint_dir, self.target_folder_id)
            if exit_code > 0:
                logger.error(f"Failed to upload checkpoint folder '{checkpoint_dir}' to Google Drive.")