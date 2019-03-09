def test_import_shmistogram():
    import shmistogram as shm
    assert '__version__' in dir(shm)

# # Not working!?
# def test_import_pydens():
#     import pydens as dens
#     assert '__version__' in dir(dens)
