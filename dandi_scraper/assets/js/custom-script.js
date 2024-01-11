
function tableDropdowncust() {
    
    console.log('adding effects')
    $("#datatable-row-ids tr>td:first-of-type").unbind()
    $("#datatable-row-ids tr>td:first-of-type").on("click",
        function() {
            row = $(this).closest('tr')
            cell_id =  $(row).find($( "td[data-dash-column='specimen_id']" )).text()
            if ($(row).hasClass("expanded")) {
                //$(".extraInfo").remove()
                $(".expanded").removeClass("expanded")
            } else {
                //$(".extraInfo").remove()
                $(".expanded").removeClass("expanded")
                $(row).addClass("expanded")
            }
            if ($(row).hasClass('expanded')) {
                //insert a new row
                $(row).after("<tr class='extraInfo'><td colspan='7'></td></tr>")
                //get the jquery object of the new row
                row_new = $(row).next().children().first()
                //wait for the new graph to be added to the DOM

                var interid = setInterval(function() {
                    var graph = $(document).find('div[id^="'+cell_id+'"]');
                    console.log("looking for graph with id: " + cell_id);
                    $(graph).appendTo($(row_new))
                }, 100)
                //clear the interval once the graph is added
                setTimeout(function() {clearInterval(interid)}, 10000) //wait 10 seconds
                

            }
        }
    )
}




window.fetch = new Proxy(window.fetch, {
    apply(fetch, that, args) {
        // Forward function call to the original fetch
        const result = fetch.apply(that, args);

        // Do whatever you want with the resulting Promise
        result.then((response) => {
            if (args[0] == '/_dash-update-component') {
                setTimeout(function() {tableDropdowncust()}, 1000)
            }})
        return result
        }
    }
    )

$(document).ready(function() {
    setTimeout(function() {tableDropdowncust()}, 1000)
})