a_kan = dict(
    verbose=True,
    tv_jac_regularization=True,
    alpha_jacobian=15.0,
    alpha_tv=0.4,
    loss_eps=0.1,
    network_type="a_kan",
    kan_layers=[3, 70, 70, 3],
    degree=84,
    init_cfg=[0, 12],
    scheduler_type="exp",
    lr=4e-4,
    epochs=1500,
    batch_size=10000
)

rand_kan = dict(
    verbose=True,
    tv_jac_regularization=True,
    alpha_jacobian=15.0,
    alpha_tv=0.4,
    loss_eps=0.1,
    network_type="rand_kan",
    kan_layers=[3, 70, 70, 3],
    degree=84,
    init_cfg=[0, 12],
    scheduler_type="onecycle",
    lr=3e-4,
    epochs=1500,
    batch_size=10000
)


kan = dict(
    verbose=True,
    tv_jac_regularization=True,
    alpha_jacobian=15.0,
    alpha_tv=0.4,
    loss_eps=0.1,
    network_type="kan",
    kan_layers=[3, 70, 70, 3],
    degree=28,
    scheduler_type="onecycle",
    lr=1e-4,
    epochs=1500,
    batch_size=10000
)


idir = dict(
    verbose=True,
    bending_regularization=True,
    network_type="SIREN",
    epochs=2500,
    scheduler_type=None,
)