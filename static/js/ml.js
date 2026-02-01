(function($) {

    $.fn.jqLoad = function(parameter) {

        var epd;
        var rightNow = new Date().getTime();

        // If no parameter, look for data-load attributes safely
        if (arguments.length === 0) {
            $("*[data-load]").each(function() {

                let loadValue = $(this).data("load");

                // Skip if missing or invalid
                if (loadValue === undefined || isNaN(loadValue)) return;

                epd = new Date(loadValue * 1000).getTime();

                if (!isNaN(epd) && epd < rightNow) {
                    $(this).hide();
                }
            });
        }

        // Arrays not supported
        else if (Array.isArray(parameter)) {
            console.log("This feature is yet to be implemented!");
        }

        // If a string timestamp
        else if (typeof parameter === "string") {

            let expiryDate = new Date(parameter).getTime();

            if (!isNaN(expiryDate) && expiryDate < rightNow) {
                this.each(function() {
                    $(this).hide();
                });
            }
        }
    };

}(jQuery));
