module dds(clk, reset, k, sampling_pulse, new_sample_ready, sample);
    input clk, reset, sampling_pulse;
    input [21:0] k;
    output new_sample_ready;
    output [15:0] sample;

    // Definition of wires
    wire[21:0] raw_addr, sigma;
    wire[9:0] rom_addr;
    wire area;
    wire[15:0] raw_data, data;

    // Instantiate d flip-flops
    assign sigma = raw_addr + k;
    dffre #(.n(22)) dffre1(.d(sigma), .en(sampling_pulse), .r(reset), .clk(clk), .q(raw_addr));
    dffre #(.n(1)) dffre2(.d(raw_addr[21]), .en(1'b1), .r(1'b0), .clk(clk), .q(area));
    dffre #(.n(16)) dffre3(.d(data), .en(sampling_pulse), .r(1'b0), .clk(clk), .q(sample));
    dffre #(.n(1)) dffre4(.d(sampling_pulse), .en(1'b1), .r(1'b0), .clk(clk), .q(new_sample_ready));

    // Instantiate address processing unit
    addr_handler addr_handler1(.raw_addr(raw_addr[20:10]), .rom_addr(rom_addr));

    // Instantiate rom
    sine_rom sine_rom1(.clk(clk), .addr(rom_addr), .dout(raw_data));

    // Instantiate data processing unit
    data_handler data_handler1(.raw_data(raw_data), .area(area), .data(data));
endmodule
