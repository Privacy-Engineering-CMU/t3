domain_node.users.create(
    **{
        "name": "Sheldon Cooper",
        "email": "sheldon@caltech.edu",
        "password": "bazinga",
        "budget": 100
    }
)

domain_node.users

ds_node = sy.login(email="sheldon@caltech.edu", password="bazinga", port=8081)

ds_node.datasets