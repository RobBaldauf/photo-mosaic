from app.services.abstract_persistence import AbstractPersistenceService


class MailService:
    def __init__(self, persistence_service: AbstractPersistenceService):
        self.persistence = persistence_service


#     # TD: segment id is already unique remove mosaic id
#     @staticmethod
#     async def send_filtered_image_email(attached_file: File, email_adr: str):
#         conf = ConnectionConfig(
#             MAIL_USERNAME=get_config().mail_username,
#             MAIL_PASSWORD=get_config().mail_password,
#             MAIL_FROM=get_config().mail_from,
#             MAIL_PORT=get_config().mail_port,
#             MAIL_SERVER=get_config().mail_server,
#             MAIL_TLS=get_config().mail_tls,
#             MAIL_SSL=get_config().mail_ssl,
#             USE_CREDENTIALS=get_config().mail_use_credentials,
#             VALIDATE_CERTS=get_config().mail_validate_certs,
#         )
#
#         message = MessageSchema(
#             subject="StaGa - Fotobox",
#             recipients=[email_adr],
#             body="Anbei Ihr verpixeltes Bild.",
#             attachments=[attached_file],
#         )
#
#         fm = FastMail(conf)
#         await fm.send_message(message)
