  config:
    install_tree:
      root: /opt/spack
      padded_length: 128

  mirrors:
    local-buildcache:
      url: oci://ghcr.io/timniederhausen/spack-buildcache
      signed: false

      access_pair:
        id_variable: GITHUB_USER
        secret_variable: GITHUB_TOKEN
