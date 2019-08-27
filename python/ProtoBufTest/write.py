import addressbook_pb2

address_book = addressbook_pb2.AddressBook()
person = address_book.people.add()
person.id = int(1234)
person.name = "Jinwoo Nam"
person.email = "junimnjw@gmail.com"
phone = person.phones.add()
phone.number = "010-1111-1111"
phone.type = addressbook_pb2.Person.HOME

try:
    f = open('myaddress', 'wb')
    f.write(address_book.SerializeToString())
    f.close()
    print("File is Written")
except IOError:
    print("IO Error Occured")
