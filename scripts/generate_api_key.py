import argparse
import sys
from datetime import datetime, timedelta

import jwt

PAYLOAD = {"app": "photo-mosaic", "id": "photo-mosaic-admin"}
LIFETIME_DAYS = 365 * 2


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--secret", help="JWT Secret for Encoding", type=str, required=True)
    args = parser.parse_args(args)
    # print("Creating JWT-Key....")
    # print(f"JWT_KEY: {create_jwt_key(PAYLOAD, args.secret)}")
    # print(f"JWT_SECRET: {args.secret}")
    print(create_jwt_key(PAYLOAD, args.secret))


def create_jwt_key(payload: dict, secret: str) -> str:
    payload["exp"] = datetime.now() + timedelta(days=LIFETIME_DAYS)
    return str(jwt.encode(payload, secret, algorithm="HS256"))


if __name__ == "__main__":
    main(sys.argv[1:])
