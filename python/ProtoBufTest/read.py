import addressbook_pb2

try:
    f = open('myaddress', 'rb')
    address_book = addressbook_pb2.AddressBook()
    address_book.ParseFromString(f.read())
    f.close()

    for person in address_book.people:
        print(person.id)
        print(person.name)
        if person.HasField('email'):
            print("Email Address:", person.email)
except IOError:
    print('file read error')




