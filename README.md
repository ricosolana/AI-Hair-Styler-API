# AI Hair Styler API

## Getting started

Set `env['JWT_SECRET_KEY']` to something secret.

## Barbershop install tips

add to path (change the version)
- `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64`

run in Administrator CMD
- (maybe optional)
  - `pip3 install --upgrade --force-reinstall ninja`
  - `pip3 install ninja`
  - `pip3 install gdown`
  - `pip3 install scikit-image`
  - `pip3 install IPython`
  - `pip3 install opencv-python`

to run:
- `python main.py --im_path1 4.png --im_path2 66.png --im_path3 31.png --sign realistic --smooth 5`

If you get a URL error showing weights fail to download from pytorch, failed, 
just keep trying the command above.

## Subprocess spawn tips

Getting an module import error?

Change your Pycharm Python version for your main program to point to the
Python correct executable (maybe global).

## TODO
- Better JWT
  - Persistent tokens across server restarts
  - Token revocation
- HTTPS
  - Figure out a trusted CA to register with
  - Plenty of free ones
- Served image-width parameter

## Nice-to-haves
- username / time-based key 
- token revocation
- Nginx HTTPS reverse proxy
  - Using a raspberry pi as the primary relay server:
    - Flutter Client -> (https) -> RPI (Nginx)
    - RPI (Nginx) -> (ngrok http -> https tunnel) -> api backends (any pc can run api and register with pi server as a device)
  - With this reverse proxy, paths can be isolated and simplified
    - Barber API does not need to worry about authentication
    - The nginx can ensure JWT authentication through another proxy...

## Self signed certificates
- Creating certs
  - https://mariadb.com/docs/server/security/data-in-transit-encryption/create-self-signed-certificates-keys-openssl/

- Trusting CA Cert
  - Firefox
    - https://javorszky.co.uk/2019/11/06/get-firefox-to-trust-your-self-signed-certificates/

