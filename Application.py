#!/usr/bin/env python
from app import create_app

application = create_app()

if __name__ == '__main__':
    application.run(host='localhost', port=8080)