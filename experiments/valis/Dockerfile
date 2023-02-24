#FROM ubuntu:bionic
FROM ubuntu:20.04

ARG PYTHON_VERSION=3.8

LABEL maintainer="arthurelsk@gmail.com"

SHELL ["/bin/bash", "-c"]

# for installing tzdata see: https://stackoverflow.com/a/58264927/4521646
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        $( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "python${PYTHON_VERSION%%.*}-distutils" ) \
        python${PYTHON_VERSION%%.*}-tk \
        build-essential \
        cmake \
        wget \
        unzip \
        git \
        ninja-build \
        ca-certificates && \

# Install python dependencies
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \

# Set the default python and install PIP packages
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \

# Disable cache
    pip config set global.cache-dir false && \
    pip install "pip>20.1" -U  && \

# Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install meson to build libvips
RUN pip install meson

# Install libvips
RUN apt-get update && \
    apt-get install --no-install-recommends -y libvips

# Install libvips dependencies
RUN apt-get install --no-install-recommends -y \
    glib-2.0-dev \
    libexpat-dev \
    librsvg2-dev \
    libpng-dev \
    libjpeg-turbo8-dev \
    libtiff-dev \
    libexif-dev \
    liblcms2-dev \
    libheif-dev \
    liborc-dev \
    libgirepository1.0-dev \
    libopenslide-dev


# Install libvips from source to get latest version
ENV LD_LIBRARY_PATH /usr/local/lib
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig

ARG VIPS_VERSION=8.14.1
ARG VIPS_URL=https://github.com/libvips/libvips/releases/download


# build the head of the stable 8.14 branch
RUN wget ${VIPS_URL}/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.xz --no-check-certificate \
	&& tar xf vips-${VIPS_VERSION}.tar.xz \
	&& cd vips-${VIPS_VERSION} \
	&& meson build --buildtype=release --libdir=lib \
	&& cd build \
	&& ninja \
	&& ninja install

RUN rm vips-${VIPS_VERSION}.tar.xz
RUN rm -r vips-${VIPS_VERSION}

# # Install OpenJDK-11-JRE
# RUN apt-get update && \
#     apt-get install -y \
#         maven \
#         openjdk-11-jre && \
#     apt-get clean;
    
# # Fix certificate issues
# RUN apt-get update && \
#     apt-get install ca-certificates-java && \
#     apt-get install -y --no-install-recommends libvips && \
#     apt-get clean && \
#     update-ca-certificates -f;

# ENV JAVA_HOME="/usr/libexec/java_home"

# Install VALIS
RUN pip install pyvips && \
    pip install valis-wsi

COPY ./ /tmp/BIRL/

RUN \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        openslide-tools && \
# Install BIRL
    # v46 crashes openslide-python install
    pip install "setuptools<46" -U && \
    pip install ./tmp/BIRL --no-cache-dir && \
    python -c "import birl ; print(birl.__version__)" && \
# Fix opencv-contrib-python issue for SIFT and BRISK
    pip uninstall -y opencv-contrib-python && \
    pip install -U opencv-contrib-python==4.7.0.68 && \
# Cleaning
    apt-get remove -y wget build-essential ninja-build && \
    apt-get autoremove -y && \
    # apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
    rm -r /tmp/BIRL

# Install other non-Python dependencies
RUN apt-get update && \
    apt-get install -y \
        maven \
        openjdk-11-jre

# Fix missing import
RUN pip install pyparsing