use crate::Yolo;
pub mod http {
    use std::fs::File;
    use std::io::copy;
    pub fn get_image(url: &str) -> anyhow::Result<()> {
        let resp = reqwest::blocking::get(url)?.bytes()?;
        let mut image = File::create("image.jpeg")?;
        copy(&mut resp.as_ref(), &mut image)?;
        Ok(())
    }
    pub fn get_image_as_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
        Ok(reqwest::blocking::get(url)?.bytes()?.into())
    }
}

pub mod socket {

    use std::{
        io::Read,
        net::{TcpListener, TcpStream},
        thread,
    };

    use crate::reco::Yolo;

    pub struct My_socket {
        listener: Box<TcpListener>,
    }

    impl My_socket {
        pub fn new(addr: &str) -> anyhow::Result<Self> {
            let listener = Box::new(TcpListener::bind(addr)?);
            Ok(My_socket { listener })
        }

        pub fn start(&self) -> anyhow::Result<()> {
            for stream in self.listener.incoming() {
                if let Ok(client) = stream {
                    println!("New socket connected! {:?}", client.peer_addr()?);
                    let _ = thread::spawn(move || My_socket::stream_handler(client)).join();
                }
            }
            Ok(())
        }
        fn stream_handler(mut client: TcpStream) {
            let mut buf = vec![0; 1024];
            let mut frame = Vec::<u8>::new();
            let yolo = Yolo::new("yolov5s.onnx", "classes.txt").unwrap();
            loop {
                if let Ok(n) = client.read(&mut buf) {
                    if n == 0 {
                        return; // closed
                    }

                    let data = &buf[..n];
                    if data.starts_with(b"IMAGE") {
                        println!("Frame start");
                        frame.extend_from_slice(&data[5..]);
                    } else if data.ends_with(b"END") {
                        frame.extend_from_slice(&data[..data.len() - 3]);
                        yolo.detect_image_from_bytes(frame.clone().into(), 1)
                            .unwrap();
                        frame.clear();
                        println!("Frame end");
                    } else {
                        frame.extend_from_slice(data);
                    }

                    // println!("start:\n{:?}\nend\n\n\n\n\n", buf);
                } else {
                    return; // error
                }
            }
        }
    }
}
