use std::{
    cell::RefCell,
    fs::File,
    io::{BufRead, BufReader},
};

use opencv::{self as cv, prelude::*};

const INPUT_WIDTH: i32 = 640;
const INPUT_HEIGHT: i32 = 640;
const SCORE_THRESHOLD: f32 = 0.2;
const NMS_THRESHOLD: f32 = 0.4;
const CONFIDENCE_THRESHOLD: f32 = 0.4;

#[derive(Debug)]
pub struct Detection {
    pub class_name: String,
    pub position: cv::core::Rect,
    pub confidence: f32,
}

pub struct Yolo {
    net: RefCell<cv::dnn::Net>,
    class_list: Vec<String>,
}

impl Yolo {
    // pub
    pub fn new(model_file: &str, class_file: &str) -> anyhow::Result<Self> {
        // load_class_list
        let mut class_list = Vec::new();
        let file = File::open(class_file)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if let Ok(class) = line {
                class_list.push(class);
            } else {
                break;
            }
        }

        // load_net_model
        let net = RefCell::new(cv::dnn::read_net_from_onnx(model_file)?);

        Ok(Yolo { net, class_list })
    }

    pub fn detect(&self, image: &cv::core::Mat) -> anyhow::Result<Vec<Detection>> {
        // prehandle
        let input_image = Self::image_format(image)?;
        let mut blob = cv::dnn::blob_from_image(
            &input_image,
            1. / 255.,
            cv::core::Size::new(INPUT_WIDTH, INPUT_HEIGHT),
            cv::core::Scalar::default(),
            true,
            false,
            cv::core::CV_32F,
        )?;

        let mut net = self.net.borrow_mut();

        net.set_input(&mut blob, "", 1.0, cv::core::Scalar::default())?;

        // forward
        let mut forward_output = cv::types::VectorOfMat::new();

        let out_blob_names = net.get_unconnected_out_layers_names()?;
        net.forward(&mut forward_output, &out_blob_names)?;

        // handle forward result
        let group_length = self.class_list.len() + 5;
        // println!("{}", group_length);

        let x_factor = input_image.cols() as f32 / INPUT_WIDTH as f32;
        let y_factor = input_image.rows() as f32 / INPUT_HEIGHT as f32;
        // println!("x_factor: {}  y_factor: {}", x_factor, y_factor);

        let mut binding = forward_output.get(0)?;
        let forward_data = binding.data_typed_mut::<f32>()?;

        let mut class_ids = cv::core::Vector::<i32>::new();
        let mut confidences = cv::core::Vector::<f32>::new();
        let mut positions = cv::core::Vector::<cv::core::Rect>::new();

        for i in 0..25200 {
            let confidence = forward_data[group_length * i + 4];
            if confidence >= CONFIDENCE_THRESHOLD {
                // refactor
                // let mut class_score = forward_data[group_length * i + 5];
                // let score = unsafe {
                //     cv::core::Mat::new_rows_cols_with_data(
                //         1,
                //         self.class_list.len() as i32,
                //         cv::core::CV_32FC1,
                //         &mut class_score as *mut f32 as *mut std::ffi::c_void,
                //         cv::core::Mat_AUTO_STEP,
                //     )?
                // };

                // let mut class_id = cv::core::Point::default();
                // let mut max_class_score: f64 = 0.0;
                // cv::core::min_max_loc(
                //     &score,
                //     None,
                //     Some(&mut max_class_score),
                //     None,
                //     Some(&mut class_id),
                //     &cv::core::no_array(),
                // )?;

                let mut class_id: usize = 0;
                let mut max_class_score: f32 = 0.0;
                for j in (group_length * i + 5)..(group_length * (i + 1)) {
                    if forward_data[j] > max_class_score {
                        class_id = j - (group_length * i + 5);
                        max_class_score = forward_data[j];
                    }
                }

                if max_class_score > SCORE_THRESHOLD {
                    confidences.push(confidence);
                    class_ids.push(class_id as i32);

                    let x = forward_data[group_length * i + 0];
                    let y = forward_data[group_length * i + 1];
                    let w = forward_data[group_length * i + 2];
                    let h = forward_data[group_length * i + 3];

                    let left = ((x - 0.5 * w) * x_factor) as i32;
                    let top = ((y - 0.5 * h) * y_factor) as i32;
                    let width = (w * x_factor) as i32;
                    let height = (h * y_factor) as i32;
                    positions.push(cv::core::Rect::new(left, top, width, height));
                }
            }
        }

        // handle possibly repeat
        let mut nms_result = cv::core::Vector::<i32>::new();
        cv::dnn::nms_boxes(
            &positions,
            &confidences,
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            &mut nms_result,
            1.,
            0,
        )?;
        let mut output = Vec::<Detection>::new();
        for i in 0..nms_result.len() {
            let idx = nms_result.get(i)? as usize;
            let class_id = class_ids.get(idx)? as usize;
            // println!("class_id: {}\n", class_id);
            let class_name = self.class_list[class_id].clone();

            output.push(Detection {
                class_name,
                confidence: confidences.get(idx)?,
                position: positions.get(idx)?,
            });
        }

        Ok(output)
    }

    pub fn detect_image(&self, image: &str, ms: i32) -> anyhow::Result<()> {
        cv::highgui::named_window("detect_image", cv::highgui::WINDOW_FULLSCREEN)?;
        let mut image_mat = cv::imgcodecs::imread(image, cv::imgcodecs::IMREAD_COLOR)?;
        let output = self.detect(&image_mat)?;

        for detection in output {
            // println!("{:#?}", detection);
            // print!("{:?}", detection._box);
            cv::imgproc::rectangle(
                &mut image_mat,
                detection.position,
                cv::core::Scalar::new(0., 0., 255., 0.),
                5,
                8,
                0,
            )?;

            cv::imgproc::put_text(
                &mut image_mat,
                detection.class_name.as_str(),
                cv::core::Point::new(detection.position.x, detection.position.y - 10),
                cv::imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::core::Scalar::new(255., 0., 0., 0.),
                2,
                8,
                false,
            )?;
        }

        cv::highgui::imshow("detect_image", &image_mat)?;
        let _ = cv::highgui::wait_key(ms);

        Ok(())
    }

    pub fn detect_image_from_bytes(
        &self,
        image_stream: cv::core::Vector<u8>,
        ms: i32,
    ) -> anyhow::Result<()> {
        cv::highgui::named_window("detect_image", cv::highgui::WINDOW_FULLSCREEN)?;
        let mut image_mat = cv::imgcodecs::imdecode(&image_stream, cv::imgcodecs::IMREAD_COLOR)?;
        let output = self.detect(&image_mat)?;

        for detection in output {
            cv::imgproc::rectangle(
                &mut image_mat,
                detection.position,
                cv::core::Scalar::new(0., 0., 255., 0.),
                5,
                8,
                0,
            )?;

            cv::imgproc::put_text(
                &mut image_mat,
                detection.class_name.as_str(),
                cv::core::Point::new(detection.position.x, detection.position.y - 10),
                cv::imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::core::Scalar::new(255., 0., 0., 0.),
                2,
                8,
                false,
            )?;
        }

        cv::highgui::imshow("detect_image", &image_mat)?;
        let _ = cv::highgui::wait_key(ms);

        Ok(())
    }

    // priv
    fn image_format(image: &cv::core::Mat) -> anyhow::Result<cv::core::Mat> {
        let cols = image.cols();
        let rows = image.rows();
        let max = std::cmp::max(cols, rows);

        let result = cv::core::Mat::zeros(max, max, cv::core::CV_8UC3)?;
        let roi = cv::core::Rect::new(0, 0, cols, rows);
        let roi_mat_exp = result.apply_1(roi)?;
        let mut roi_mat = roi_mat_exp.to_mat()?;
        image.copy_to(&mut roi_mat)?;

        Ok(roi_mat)
    }
}
