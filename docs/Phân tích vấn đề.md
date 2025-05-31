# Mục lục
1. [Đề tài](#De_tai)
2. [Mục tiêu](#Muc_tieu)
3. [Input](#Input)
4. [Các bước thực hiện](#Steps)
	1. [Dự đoán giá sản phẩm](#Goals1)
	2. [Đường đi giảm thiểu chi phí vận chuyển](#Goals2)
	3. [Sơ đồ các bước thực hiện](#Flowchart)
5. [Các thách thức](#Challenge)

<div id='De_tai'/>

# Đề tài

Tối ưu chuỗi cung ứng nông nghiệp (Agri Data Analytics)

🎯 Mục tiêu: <br>
Phân tích dữ liệu chuỗi cung ứng nông sản để tối ưu giá bán, tồn kho, phân phối. <br>
🔧 Công nghệ:<br>
Forecasting: ARIMA, LSTM<br>
Optimization: Linear programming<br>
Dashboard: nông dân → người bán → người tiêu dùng<br>
💡 Ứng dụng:<br>
Dễ gắn với dữ liệu Việt Nam (gạo, trái cây, cà phê)<br>
Dễ cộng tác với doanh nghiệp thực tế<br>

<div id='Muc_tieu'/>

# Mục tiêu
1. Dự đoán giá sản phẩm
2. Tìm đường đi giảm thiểu chi phí vận chuyển

<div id='Input'/>

# Input
Dữ liệu giá nông sản của Việt Nam từ 1/1/2020 đến ngày 19/5/2025 đối với 3 loại là cà phê, gạo lúa và rau quả.

Nguồn: [Giá thị trường nông sản](https://thitruongnongsan.gov.vn/vn/nguonwmy.aspx)

Dữ liệu được lấy về sẽ có dạng html, chuyển về dạng bảng dưới dạng dataframe bằng lệnh read_html của thư viện pandas. Tổng cộng sẽ có 3 phần: "Cà phê", "gạo, lúa", "rau, quả" và cần được tổng hợp lại thành bảng chung để có thể tuỳ biến.

Dataset gồm 51488 dòng, 8 thuộc tính:
1. Tên mặt hàng: Gồm 348 mặt hàng khác nhau gồm loại gạo, lúa, loại rau và quả. Ví dụ: Cà phê Robusta nhân xô, ST25, chanh,.... <br>
2. Thị trường: Chỉ có 22 tỉnh, bao gồm: Đắk Lắk, Đắk Nông, Gia Lai, Kon Tum, Lâm Đồng, Hồ Chí Minh, Cần Thơ, Sóc Trăng, Đồng Tháp, Kiên Giang, An Giang, Tiền Giang, Trà Vinh, Hậu Giang, Cà Mau, Bạc Liêu, Thái Bình, Hà Nội, Bến Tre, Long An, Vĩnh Long và Sơn La. <br>
3. Loại giá: Có 12 loại giá, bao gồm: Thương lái thu mua, công ty thu mua, đại lý thu mua, thu mua, bán ra, xuất khẩu, tại chợ, nán lẻ, khác, thu mua tại vườn, bán buôn, vựa thu mua. <br>
4. Đơn vị tính: Có 6 đơn vị tính, bao gồm: VNđ/Kg, VNĐ/mớ, VNĐ/Chục quả, VNĐ/củ, VNĐ/Quả, VNĐ/cây. <br>
5. Loại tiền: Chỉ có 1 giá trị là VNĐ. <br>
6. Nguồn: Có 6 nguồn, bao gồm: CTV địa phương, CTV Agroinfo/ Tintaynguyen, Giồng Riềng, Bán lẻ, Long Xuyên, Tỉnh An Giang, Tri Tôn, Cờ Đỏ, Cao Lãnh, Thành phố Thái Bình, huyện Quỳnh Phụ, huyện Thái Thụy, huyện Kiến Xương, huyện Đông Hưng. <br>
7. Ngày: Thời điểm mà giá thị trường thay đổi. <br>
8. Giá: Mang miền giá trị từ 0 đến 1 335 333 333. Cần xem xét lại.

Trong đó sẽ sử dụng các thuộc tính: Tên mặt hàng, thị trường, ngày và giá (đơn vị tính VND). 

Ưu điểm: Được lấy dữ liệu thực tế và gần gũi với thị trường Việt Nam.<br>
Nhược điểm: Không thể sử dụng real time do không lấy trực tiếp dữ liệu từ CSDL.


<div id='Steps'/>

# Các bước thực hiện

<div id='Goals1'/>

## Dự đoán giá sản phẩm<a name="Goals1"></a>
1. Lập metadata: Do chưa xác định rõ ràng được nên sử dụng (các) sản phẩm nào để huấn luyện mô hình, nên cần cần lưu lại với id cụ thể. Từ đó mà tra thông tin nhanh chóng.
2. Tiền xử lý dữ liệu:  Sau khi kiểm tra sơ bộ thì tập dữ liệu thị trường nông sản có giá trị ngoại lai khi giá thấp nhất của sản phẩm là 0, và giá cao nhất là hơn 1 tỷ. Ngoài ra không có giá trị bị thiếu và 3 giá trị bị thừa. Chuyển kiểu chuỗi sang kiểu thời gian.
3. Chọn các sản phẩm muốn huấn luyện và one-hot: Do chưa xác định rõ sẽ dùng bao nhiêu sản phẩm và mô hình thế nào. Nếu như sử dụng mô hình MTS, bước one-hot có thể cần thiết.
4. Chia tập dữ liệu: Theo như dự kiến thì sẽ chia thành 3 tập train, val, test tương ứng với 70%, 20% và 10%. Chia theo kiểu stratify.
5. Chuẩn hoá dữ liệu: Do dữ liệu không đồng đều về đơn vị tính, nên sẽ sử dụng phương pháp chuẩn hoá z-score thay vì min-max.
6. Huấn luyện mô hình: Chọn mô hình cụ thể theo lý thuyết đã đưa cùng với các sản phẩm đã chọn. Trong quá trình chạy sẽ lưu lại lịch sử để có thể đánh giá mô hình.
7. Kiểm thử: Từ kết quả huấn luyện ở bước trên, đưa ra giả thuyết để có thể tinh chỉnh mô hình.
8. Tinh chỉnh: Điều chỉnh tham số, siêu tham số theo lý thuyết đã đưa ra cho đến khi ra kết quả có thể chấp nhận được.

Sau đây là một số mô hình phân tích dữ liệu thời gian chúng em tìm hiểu được (Time series model)

|**Đặc điểm**|**ARIMA**|**LSTM**|**SARIMA**|**Transformer**|**VAR**|
|---|---|---|---|---|---|
|**Trường hợp sử dụng**|- Chuỗi thời gian **đơn biến**.<br><br>- Dữ liệu có **tính dừng** hoặc có thể đạt được sau khi **sai phân**.<br><br>- Dự báo ngắn/trung hạn.- Dữ liệu **tuyến tính, ổn định**.|- Chuỗi thời gian **phi tuyến**, **dài hạn**.<br><br>- Có thể xử lý **đơn biến hoặc đa biến**.<br><br>- Phù hợp nhiều dạng dữ liệu (văn bản, giọng nói…).<br><br>- Có thể tích hợp biến ngoại sinh.|- Chuỗi đơn biến có tính **mùa vụ rõ rệt**.<br><br>- Dữ liệu có thể đạt được **tính dừng theo mùa**.<br><br>- Dự báo ngắn và trung hạn.|- Chuỗi thời gian **dài, phức tạp**.<br><br>- Cần nắm bắt các tương tác toàn cục.<br><br>- Có thể dùng cho **đơn biến/đa biến**.<br><br>- Bắt nguồn từ NLP, nay mở rộng.|- Dữ liệu **đa biến**, các biến ảnh hưởng qua lại.<br><br>- Dự báo ngắn/trung hạn.<br><br>- Phân tích **cú sốc, tác động hệ thống**.|
|**Ưu điểm**|- Cơ sở thống kê mạnh.<br><br>- Dễ diễn giải các tham số (p,d,q).- Tốt với dữ liệu tuyến tính.- Triển khai đơn giản.|- Mô hình hóa **phi tuyến mạnh**.<br><br>- Bắt phụ thuộc dài hạn tốt.<br><br>- Linh hoạt kiến trúc.<br><br>- Không cần dữ liệu có tính dừng.|- Mô hình hóa tốt **mùa vụ**.<br><br>- Mở rộng từ ARIMA.<br><br>- Diễn giải rõ (p,d,q,P,D,Q,s).|- Bắt được **phụ thuộc toàn cục** nhờ attention.<br><br>- Huấn luyện nhanh (so với RNN).<br><br>- Hiệu suất cao (state-of-the-art).|- Mô hình hóa **tương tác giữa biến**.<br><br>- Dễ kiểm định nhân quả Granger.<br><br>- Diễn giải qua hệ số.|
|**Nhược điểm**|- Giả định tuyến tính.<br><br>- Yêu cầu dữ liệu có tính dừng.<br><br>- Không tốt với phụ thuộc dài hạn.- Không tích hợp biến ngoại sinh (trừ ARIMAX).<br><br>- Chọn tham số khó.|- "Hộp đen", khó diễn giải.<br><br>- Cần nhiều dữ liệu.- Huấn luyện tốn tài nguyên.<br><br>- Dễ overfit.<br><br>- Nhiều siêu tham số.|- Cũng giả định tuyến tính.<br><br>- Cần dữ liệu dừng.<br><br>- Nhiều tham số hơn ARIMA.<br><br>- Phức tạp nếu mùa không ổn định.|- "Hộp đen", rất khó diễn giải.<br><br>- Cần lượng dữ liệu và tài nguyên lớn.<br><br>- Khó triển khai, tinh chỉnh.<br><br>- Không hiệu quả với chuỗi ngắn.|- Giả định tuyến tính.<br><br>- Yêu cầu tất cả chuỗi phải có tính dừng.<br><br>- Số tham số tăng nhanh.<br><br>- Khó diễn giải khi có nhiều biến.|
|**Ghi chú**|- p: bậc AR<br><br>- d: sai phân<br><br>- q: MA<br><br>- Dùng ADF, KPSS để kiểm tra tính dừng- Chọn p,q bằng ACF, PACF|- Là một dạng RNN.<br><br>- Cần chuẩn hóa dữ liệu.- LSTM giảm hiện tượng vanishing gradient so với RNN.|- Tham số: (p,d,q), (P,D,Q,s)- s là chu kỳ mùa vụ.- Là mở rộng của ARIMA.|- Sử dụng **self-attention**.<br><br>- Cần **positional encoding** để học vị trí.<br><br>- Kiến trúc phức tạp.|- Biến trong hệ đều là **nội sinh**.<br><br>- Có thể mở rộng thành SVAR để mô hình hóa nhân quả.<br><br>- Chọn độ trễ (lag) rất quan trọng.|

Sau đây là một số nhận xét sơ bộ về các mô hình:

| **Mô hình**        | **Có phù hợp không?**             | **Ghi chú**                                                                                                                                                                                                               |
| :------------------ | :---: | :--- |
| **ARIMA / SARIMA** | Không phù hợp                     | Phải xây dựng riêng từng mô hình cho từng sản phẩm, khó mở rộng với 10–20 sản phẩm. Không xử lý tốt yếu tố phi tuyến hoặc biến ngoại sinh như nguồn cung.                                                                 |
| **VAR**            | Tạm được                          | Chỉ nên dùng nếu số lượng sản phẩm nhỏ hơn 5–7 do curse of dimensionality. Phải xử lý dữ liệu thành dạng dừng, mất thời gian tiền xử lý.                                                                                  |
| **LSTM**           | **Phù hợp**                       | - Có thể xây dựng 1 mô hình tổng hợp để học các chuỗi giá nhiều sản phẩm.- Xử lý được dữ liệu phi tuyến, có yếu tố thời gian.- Có thể thêm biến như loại sản phẩm, nguồn cung làm **embedding** hoặc **biến ngoại sinh**. |
| **Transformer**    | **Rất phù hợp nếu có tài nguyên** | - Xử lý tốt dữ liệu phức tạp, dài, đa chiều.- Có thể học được **mối quan hệ giữa các sản phẩm khác nhau**.- Yêu cầu dữ liệu nhiều và tài nguyên tính toán lớn hơn LSTM.                                                   |
<div id='Goals2'/>

## Đường đi giảm thiểu chi phí vận chuyển
1. Mô hình hoá bài toán: Từ đề tài đã cho, cần xác định được các biến, hàm, ràng buộc cần và đủ để có thể giải một bài toán tối ưu.
2. Thiết lập bảng chi phí: Chi phí ở đây có thể là khoảng cách hoặc chi phí vận chuyển hàng hoá nếu tìm được. Đơn giản nhất là khoảng cách giữa 22 tỉnh đã có trong input. Có thể thực hiện song song với bước 1.
3. Giải bài toán: Chọn một phương pháp cụ thể và giải bài toán.
4. Kiểm tra tối ưu: Sử dụng các phương pháp để kiểm tra tính tối ưu của bài toán, nếu bài toán chưa được tối ưu thì sẽ phải thay đổi biến, hàm, ràng buộc,...
<div id='Flowchart'/>

## Sơ đồ các bước thực hiện<a name="Flowchart"></a>
![Hinh](https://i.ibb.co/JW1CdpkM/z6656993057334-26bdb8974f56842425052f6f5566cbef.jpg)

<div id='Challenge'/>

# Các thách thức <a name="Challenge"></a>
- Tài nguyên tính toán yếu: Tài nguyên của 2 máy lap đều tương đối yếu, nhưng có thể giải quyết bằng hiện kim.
- Chưa có kinh nghiệm lập trình tuyến tính: Tối ưu hoá nói chung là một bài toán khó, việc chưa có kinh nghiệm có thể gặp rất nhiều khó khăn trong quá trình làm.
- Chưa rõ về dashboard: Theo như yêu cầu đề tài thì có sử dụng dashboard. Tuy nhiên, dữ liệu được lấy không phải là real-time nên việc tạo dashboard chỉ có thể xem dữ liệu trong thời gian và không thể cập nhật được thêm.

