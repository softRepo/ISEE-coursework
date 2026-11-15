module data_handler(raw_data, area, data);
    // The module is for processing data

    input[15:0] raw_data;
    input area;
    output reg[15:0] data;

    always @(*) begin
        if (area)
            data = ~raw_data + 1;
        else
            data = raw_data;
    end
endmodule
