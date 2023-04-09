mod reco;
mod req;
use reco::Yolo;
use req::http::get_image_as_bytes;
use req::socket::My_socket;

fn main() -> anyhow::Result<()> {
    let my_socket = My_socket::new("192.168.31.105:8080")?;
    my_socket.start()

    // image_detect_loop()
}

fn image_detect_loop() -> anyhow::Result<()> {
    loop {
        let image_stream = get_image_as_bytes("http://192.168.31.224/")?;
        let yolo = Yolo::new("yolov5s.onnx", "classes.txt")?;
        yolo.detect_image_from_bytes(image_stream.into(), 1)?;
        let key = opencv::highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }
    }
    Ok(())
}
