import wget

url='https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0-small.zip&h=a2ef047345afd722e3d351834d8203e387f3661599f575e6272cf60821d6a21e&v=1&xid=1201cfc330&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+Subscription+Confirmed'
def main():
    filename = wget.download(url)


if __name__ == "__main__":
    main()
