# Chapter 2: Image formation, acquisition and digitization

## 1. Image formation
> Cách thức tạo ra ảnh qua thị giác của người

* The human eye is a camera

* Iris, pupil

* Cones, rods
  - Tế bào hình nón --> nhạy cảm với màu, cho phép phân biệt các màu khác nhau 
  - Tế bào hình que --> nhạy cảm với ánh sáng, phân biệt được các đối tượng trong vùng tối `

* Image formation
  - Source --> scene element --> image sys --> image plane

* Light
  - Phần nhìn thấy được của quang phổ điện từ
  - 400-700 nanometers 

## 2. Acquisition & digitization: Digital camera
> Qúa trình thu nhận ảnh để số hóa ảnh \
> Nguyên lý của camera số

* Processing pipeline
  - Camera irradiance --> camera body --> sensor chip --> RAW (dung lượng lớn) 
  - Camera irradiance --> camera body --> sensor chip --> DSP (digital signal processing) --> JPEG (dung lượng nhỏ)

* Filter
  - Bước sóng thấp  --> chỉ cho ánh sáng Blue đi qua
  - Bước sóng tb    --> Green (mắt người nhạy cảm với màu xanh lá nhất)
  - Bước sóng cao   --> Red

* Real scene --> digital image
  - Scene + light --> electric charge (continunous signal) --> number (discrete signal)
  - Digitization = Sampling (lấy mẫu) + Quantization (lượng tử hóa)
  - Lấy mẫu: quyết định mức độ chi tiết của đối tượng quan sát được
  - Lượng tử hóa: quyết định giá trị khác nhau của mỗi pixel trong ảnh 
  > ex. 8 bits --> 2^8=256 --> [0,255]
  - Độ phân giải ~ số lượng sensor

## 3. Color
> Không gian màu

* Color spaces
  - RGB
  > Không tách được độ sáng, màu sắc
  - HSV
  > Hue (màu, [0,359], góc), Saturation (độ bão hòa, [0,1], bán kính), Value ([0,255])
  - (YUV,LUV)
  - XYZ

## 4. Digital image representation & formats
> Một số định dạng khi lưu trữ
